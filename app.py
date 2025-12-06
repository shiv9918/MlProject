from flask import Flask, request, render_template
import os
import sys

# Initialize Flask app
application = Flask(__name__)
app = application

# Verify required files exist on startup
def check_required_files():
    """Check if required model and preprocessor files exist"""
    model_path = "artifacts/model.pkl"
    preprocessor_path = "artifacts/preprocessor.pkl"
    
    if not os.path.exists(model_path):
        print(f"WARNING: Model file not found at {model_path}", file=sys.stderr)
    if not os.path.exists(preprocessor_path):
        print(f"WARNING: Preprocessor file not found at {preprocessor_path}", file=sys.stderr)
    
    return os.path.exists(model_path) and os.path.exists(preprocessor_path)

# Check files on application startup
check_required_files()

# Health check route for Elastic Beanstalk
@app.route('/health')
def health():
    return {'status': 'healthy'}, 200

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')   # <-- AWS default page hata diya


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    try:
        from src.pipeline.predict_pipeline import CustomData, PredictPipeline
        
        if request.method == 'GET':
            return render_template('home.html')

        else:
            # Get form data
            gender = request.form.get('gender')
            race_ethnicity = request.form.get('race_ethnicity')
            parental_level_of_education = request.form.get('parental_level_of_education')
            lunch = request.form.get('lunch')
            test_preparation_course = request.form.get('test_preparation_course')
            reading_score = request.form.get('reading_score')
            writing_score = request.form.get('writing_score')

            # Validate all fields
            if not all([
                gender, race_ethnicity, parental_level_of_education,
                lunch, test_preparation_course, reading_score, writing_score
            ]):
                return render_template('home.html', error="Please fill in all fields.")

            data = CustomData(
                gender=gender,
                race_ethnicity=race_ethnicity,
                parental_level_of_education=parental_level_of_education,
                lunch=lunch,
                test_preparation_course=test_preparation_course,
                reading_score=float(reading_score),
                writing_score=float(writing_score)
            )

            pred_df = data.get_data_as_data_frame()
            print(pred_df)

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            return render_template('home.html', results=results[0])

    except ImportError as e:
        return f"Import error: {str(e)}", 500
    except Exception as e:
        import traceback
        error_msg = f"An error occurred: {str(e)}\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        return render_template('home.html', error=f"An error occurred: {str(e)}")


# --------- Only for local debugging, not used by EB ----------
if __name__ == "__main__":
    # For local run: python application.py
    app.run(host="0.0.0.0", port=5000, debug=True)
