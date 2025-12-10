from flask import Flask, render_template, request, jsonify
from src.pipeline.prediction_pipeline import StoringData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def homepage():
    return render_template('index.html') 

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html') 
    else:
        data = StoringData(
            hours_studied= float(request.form.get('Hours_Studied')),
            previous_score= float(request.form.get('Previous_Scores')),
            ec_activity= request.form.get('Extracurricular_Activities'),
            sleep_hours= float(request.form.get('Sleep_Hours')),
            question_paper_practiced= float(request.form.get('Sample_Question_Papers_Practiced')),
        )
    

        final_df = data.get_data_as_dataframe()

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(final_df)

        return render_template('form.html', final_result=results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)