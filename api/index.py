from flask import Flask, request, jsonify, send_from_directory
import pickle  # Replace with your ML model import
import pandas as pd
from flask_cors import CORS,cross_origin



course = ['Machine Learning', 'Analyst', 'Software Development', 'Web-Dev']
performance = ['Excellent', 'Good', 'Average', 'Bad']
placement = ['Will', 'Will Not']
skills=['Android', 'ML', 'DL', 'C', 'UI/UX', 'Backend', 'Frontend','Full-Stack', 'Python', 'C++']
columns=['UID', 'Name', 'Sex', 'Age','10th', '12th', 'Sem1', 'Sem2','Sem3', 'Sem4', 'Sem5', 'Sem6', 'Sem7', 'Current CGPA', 'AMCAT','Skill1', 'Skill2', 'Skill3', 'Skill4', 'Avg. Attendance']

mapping = {value: index for index, value in enumerate(skills)}

app = Flask(__name__,static_folder="../dist",static_url_path="")
CORS(app)

@app.route("/")
@cross_origin()
def index():
  return send_from_directory(app.static_folder,"index.html")
@app.errorhandler(404)
@cross_origin()
def not_found(e):
  return send_from_directory(app.static_folder,'index.html')

@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    data=request.get_json()
    data["Skills"]=[skill.strip() for skill in data["Skills"].split(",")]
    data["SGPA"]=[cgpa.strip() for cgpa in data["SGPA"].split(",")]
    input=[]
    for sub in data.values():
        if type(sub)==list:
            for x in sub:
                input.append(x)
        else:
            input.append(sub)
    user_input = pd.DataFrame([input], columns=columns)
    input_data = user_input.drop(['UID', 'Name', 'Sex', 'Age'], axis=1)

    skill_mapping = {value: index for index, value in enumerate(skills)}


    input_data['Skill1'] = input_data['Skill1'].map(skill_mapping)
    input_data['Skill2'] = input_data['Skill2'].map(skill_mapping)
    input_data['Skill3'] = input_data['Skill3'].map(skill_mapping)
    input_data['Skill4'] = input_data['Skill4'].map(skill_mapping)
    
    # input_data = minmax.transform(input_data)
    model1 = pickle.load(open('course_assigned_model.pkl', 'rb'))
    model2 = pickle.load(open('performance_model.pkl', 'rb'))
    model3 = pickle.load(open('placed_status_model.pkl', 'rb'))
    

    username = user_input['Name'][0]
    recommended_course = course[model1.predict(input_data)[0] - 1]
    user_performance = performance[model2.predict(input_data)[0] - 1]
    placement_status = placement[model3.predict(input_data)[0] - 1]
    message = (f'Hi, {username}, I think you can study {recommended_course} course '
                f'and you are {user_performance} in performance. '
                f'Also, you {placement_status} get placed easily.')
    print(message)
    return  jsonify({'username': username,
                     'prediction': message,
                     'recommended_course':recommended_course,
                     'user_performance':user_performance,
                     'placement_status':placement_status})


if __name__ == "__main__":
  app.run()
