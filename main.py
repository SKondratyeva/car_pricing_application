from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from functions import process_input
import pickle
from optional_features import optional_features


app = FastAPI()
app.mount("/app/templates", StaticFiles(directory="templates"), name="templates")
templates = Jinja2Templates(directory="templates")

model_file = open('model/best_model_xgb_reg.pkl', 'rb')
model = pickle.load(model_file)

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict")
async def predict(request: Request,
    MAKE_LABEL: str = Form(...),
    CAR_MODEL: str = Form(...),
    FUEL_TYPE_ID: str = Form(None),
    MANUFACTURE_YEAR: str = Form(None),
    CUBIC_CAPACITY: str = Form(None),
    POWER: str = Form(...),
    MILEAGE: str = Form(...),
    TRANSMISSION_ID: str = Form(...),
    EMISSION_CLASS_ID: str = Form(None),
    DRIVE_ID: str = Form(None),
    FUEL_CONSUMPTION_URBAN: str = Form(optional_features['FUEL_CONSUMPTION_URBAN']),
    FUEL_CONSUMPTION_EXTRA_URBAN: str = Form(optional_features['FUEL_CONSUMPTION_EXTRA_URBAN']),
    FUEL_CONSUMPTION_COMBINED: str = Form(optional_features['FUEL_CONSUMPTION_COMBINED']),
    WEIGHT: str = Form(None),
    NUMBER_OF_GEARS: str = Form(None),
    CARBON_DIOXIDE_EMISSION: str = Form(optional_features['CARBON_DIOXIDE_EMISSION']),
    INTERIOR_MATERIAL: str = Form(None),
    SELLER_COUNTRY: str = Form(None)
):

    dict_input =  {
        "MAKE_LABEL": MAKE_LABEL,
        "MODEL": CAR_MODEL,
        "FUEL_TYPE_ID": FUEL_TYPE_ID,
        "MANUFACTURE_YEAR": MANUFACTURE_YEAR,
        "CUBIC_CAPACITY": CUBIC_CAPACITY,
        "POWER": POWER,
        "MILEAGE": MILEAGE,
        "TRANSMISSION_ID": TRANSMISSION_ID,
        "EMISSION_CLASS_ID": EMISSION_CLASS_ID,
        "DRIVE_ID": DRIVE_ID,
        "FUEL_CONSUMPTION_URBAN": FUEL_CONSUMPTION_URBAN,
        "FUEL_CONSUMPTION_EXTRA_URBAN": FUEL_CONSUMPTION_EXTRA_URBAN,
        "FUEL_CONSUMPTION_COMBINED": FUEL_CONSUMPTION_COMBINED,
        "WEIGHT": WEIGHT,
        "NUMBER_OF_GEARS": NUMBER_OF_GEARS,
        "CARBON_DIOXIDE_EMISSION": CARBON_DIOXIDE_EMISSION,
        "INTERIOR_MATERIAL": INTERIOR_MATERIAL,
        "SELLER_COUNTRY": SELLER_COUNTRY
    }

    df = process_input(dict_input)

    price_predicted = model.predict(df)

    price_predicted = int(price_predicted[0])

    return templates.TemplateResponse("response_page.html", {"prediction": price_predicted,
                                                              "request": request})
