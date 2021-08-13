import numpy as np
import xgboost as xgb

target_cols = ['ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1',
               'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
               'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1', 'ind_pres_fin_ult1',
               'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1', 'ind_nomina_ult1',
               'ind_nom_pens_ult1', 'ind_recibo_ult1']
target_cols = np.array(target_cols)
default_age = round(2 / 7, 4)
income = round(101850 / 1500000, 6)
age_index = 15
seniority_index = 16
income_index = 17
gender_index = 1
activity_index = 11


def get_normalized_age(age):
    mean_age = 40.
    min_age = 20.
    max_age = 90.
    range_age = max_age - min_age
    if age == 'NA' or age == '':
        age = mean_age
    else:
        age = float(age)
        if age < min_age:
            age = min_age
        elif age > max_age:
            age = max_age
    return round((age - min_age) / range_age, 4)


def get_normalized_seniority(seniority):
    min_value = 0.
    max_value = 256.
    range_value = max_value - min_value
    missing_value = 0.
    if seniority == 'NA' or seniority == '':
        seniority = missing_value
    else:
        seniority = float(seniority)
        if seniority < min_value:
            seniority = min_value
        elif seniority > max_value:
            seniority = max_value
    return round((seniority - min_value) / range_value, 4)


def get_normalized_income(annual_income):
    min_value = 0.
    max_value = 1500000.
    range_value = max_value - min_value
    missing_value = 101850.
    if annual_income == 'NA' or annual_income == '':
        annual_income = missing_value
    else:
        annual_income = float(annual_income)
        if annual_income < min_value:
            annual_income = min_value
        elif annual_income > max_value:
            annual_income = max_value
    return round((annual_income - min_value) / range_value, 6)


def get_gender_mapped_value(gender):
    if gender not in ['', 'NA']:
        if gender == 'Male':
            return 1
        if gender == "Female":
            return 0
    return 2


def get_activity_mapped_value(is_active):
    if is_active not in ['', 'NA']:
        if is_active == 1:
            return 1
        if is_active == 0:
            return 0
    return 2


# return a list with the default value for each attribute
def get_default_customer_info():
    return [0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 6, default_age, 0, income] + ([0] * 22)


# cast the list to xgb.DMatrix to fit with the model
def prepare_data_for_prediction(customer_info):
    model_input = [customer_info]
    model_input = np.array(model_input)
    model_input = xgb.DMatrix(model_input)
    return model_input


# update each of the 5 parameters age,annual_income,seniority,gender and is_active , and ignore the missing values
def set_parameters_if_any(customer_info, customer_default_info):
    if 'age' in customer_info and customer_info['age'] is not None:
        customer_default_info[age_index] = get_normalized_age(customer_info['age'])
    if 'income' in customer_info and customer_info['income'] is not None:
        customer_default_info[income_index] = get_normalized_income(customer_info['income'])
    if 'seniority' in customer_info and customer_info['seniority'] is not None:
        customer_default_info[seniority_index] = get_normalized_seniority(customer_info['seniority'])
    if 'gender' in customer_info and customer_info['gender'] is not None:
        customer_default_info[gender_index] = get_gender_mapped_value(customer_info['gender'])
    if 'is_active' in customer_info and customer_info['is_active'] is not None:
        customer_default_info[activity_index] = get_activity_mapped_value(customer_info['is_active'])


# for model loading and prediction
class Recommender:
    def __init__(self):
        self.model = None

    def load_model(self):
        self.model = xgb.Booster()
        self.model.load_model("model.txt")

    def predict(self, model_input):
        predictions = self.model.predict(model_input)
        predictions = np.argsort(predictions, axis=1)
        final_predictions = [" ".join(list(target_cols[prediction])) for prediction in predictions]
        return final_predictions

    def recommend(self, customer_info):
        customer_default_info = get_default_customer_info()
        set_parameters_if_any(customer_info, customer_default_info)
        model_input = prepare_data_for_prediction(customer_default_info)
        recommended_products = self.predict(model_input)
        return recommended_products
