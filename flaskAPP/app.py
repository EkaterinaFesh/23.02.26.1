#import
import pickle

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from flask import (Flask,#сервер
                   render_template,#отображение шаблонов
                   request,
                   jsonify) #обработка запросов

print("import success")
#load models
contact_LE = pickle.load(open("/home/ekast/23.02.26.1/flaskAPP/tech_models/contact_LE.pkl", "rb"))
default_LE = pickle.load(open("/home/ekast/23.02.26.1/flaskAPP/tech_models/default_LE.pkl", "rb"))
education_LE = pickle.load(open("/home/ekast/23.02.26.1/flaskAPP/tech_models/education_LE.pkl", "rb"))
housing_LE = pickle.load(open("/home/ekast/23.02.26.1/flaskAPP/tech_models/housing_LE.pkl", "rb"))
job_LE =pickle.load(open("/home/ekast/23.02.26.1/flaskAPP/tech_models/job_LE.pkl", "rb"))
loan_LE = pickle.load(open("/home/ekast/23.02.26.1/flaskAPP/tech_models/loan_LE.pkl", "rb"))
marital_LE = pickle.load(open("/home/ekast/23.02.26.1/flaskAPP/tech_models/marital_LE.pkl", "rb"))
month_LE = pickle.load(open("/home/ekast/23.02.26.1/flaskAPP/tech_models/month_LE.pkl", "rb"))
poutcome_LE = pickle.load(open("/home/ekast/23.02.26.1/flaskAPP/tech_models/poutcome_LE.pkl", "rb"))
y_LE = pickle.load(open("/home/ekast/23.02.26.1/flaskAPP/tech_models/y_LE.pkl", "rb"))
num_scaler = pickle.load(open("/home/ekast/23.02.26.1/flaskAPP/tech_models/num_scaler.pkl", "rb"))
KNN = pickle.load(open("/home/ekast/23.02.26.1/flaskAPP/ml_models/kNN.pkl", "rb"))
print("models loeaded success")
#app
app = Flask(__name__)

#route 
@app.route("/", methods=["POST", "GET"])
def main():
    #отображение формы анкеты по умолчанию
    if request.method == "GET":
        return render_template("main.html")
    #если пользователь отправил форму с веб страницы
    if request.method == "POST":
        print("Страница в разработке")
        #get data from form
        contact = request.form["contact"]
        default = request.form["default"]
        education = request.form["education"]
        housing  = request.form["housing"]
        job = request.form["job"]
        loan = request.form["loan"]
        marital = request.form["marital"]
        month = request.form["month"]
        poutcome = request.form["poutcome"]
        age = request.form["age"]
        balance = request.form["balance"]
        day = request.form["day"]
        duration = request.form["duration"]
        campaign = request.form["campaign"]
        pdays = request.form["pdays"]
        previous = request.form["previous"]


        #get preprocessing

        #categorical
        x_cat_list = [contact, #список переменных
                        default,
                        education,
                        housing,
                        job, 
                        loan,
                        marital,
                        month,
                        poutcome ]
        le_list = [contact_LE,  #список кодировщиков
                        default_LE,
                        education_LE,
                        housing_LE,
                        job_LE, 
                        loan_LE,
                        marital_LE,
                        month_LE,
                        poutcome_LE ]
        x_le_list = [] #помещать закодированные признаки

        for i in range(len(x_cat_list)):
            x_cat = le_list[i].transform([x_cat_list[i]])[0] #0 чтобы забрать значение из массива
            x_le_list.append(x_cat)
        print("x_le_list", x_le_list)
        
        #num
        x_num = [age,
                balance,
                day,
                duration,
                campaign,
                pdays,
                previous]
        #Собираем общий Х
        X = []
        X.extend(x_le_list)
        X.extend(x_num)
        #Scaler
        X_scaled = num_scaler.transform([X])
        print("X_scaled:",X_scaled)
        
        #predict
        prediction = KNN.predict(X_scaled)
       
        #return result
        if prediction == 0:
            result = "Извините мы не можем выдавать вам кредит"
        else:
            result = "Поздравляем кредит одобрен!"
        return render_template ("result.html", result = result)
@app.route("/api/v1/add_message/", methods = ["POST", "GET"])
def api_message():
    get_message_x = request.json
    X = get_message_x["X_scaled"]
    prediction = KNN.predict(X)
    if prediction == 0:
        result = "Извините мы не можем выдавать вам кредит"
    else:
        result = "Поздравляем кредит одобрен!"
    return jsonify(str(result))
@app.route("/api/v2/add_message_v2/", methods = ["POST", "GET"])
def api_message_v2():
   
    return "api в разработке"
if __name__ == "__main__":
    app.run(debug=True)