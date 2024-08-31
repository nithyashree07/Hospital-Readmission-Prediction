from flask import Flask, request, render_template, redirect, url_for
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

def load_model(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except EOFError:
        print(f"Error: The file {path} is empty or corrupted.")
    except Exception as e:
        print(f"Error loading model from {path}: {e}")
    return None

# Load models and scaler
# mlp_model = load_model('models/models/mlp_model.pkl')
rf_model = load_model('models/models/balanced_rf_model.pkl')
scaler = load_model('models/models/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/datainput')
def datainput():
    return render_template('datainput.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    try:
        # Extract feature values from the form data
        feature_dict = {
            'time_in_hospital': [float(data.get('time-in-hospital', 0))],
            'n_lab_procedures': [float(data.get('num-lab-procedures', 0))],
            'n_procedures': [float(data.get('num-procedures', 0))],
            'n_medications': [float(data.get('num-medications', 0))],
            'n_outpatient': [float(data.get('num-outpatient-visits', 0))],
            'n_inpatient': [float(data.get('num-inpatient-visits', 0))],
            'n_emergency': [float(data.get('num-emergency-visits', 0))],
            'age_[50-60)': [0],
            'age_[60-70)': [0],
            'age_[70-80)': [0],
            'age_[80-90)': [0],
            'age_[90-100)': [0],
            'medical_specialty_Emergency/Trauma': [0],
            'medical_specialty_Family/GeneralPractice': [0],
            'medical_specialty_InternalMedicine': [0],
            'medical_specialty_Missing': [0],
            'medical_specialty_Other': [0],
            'medical_specialty_Surgery': [0],
            'diag_1_Diabetes': [0],
            'diag_1_Digestive': [0],
            'diag_1_Injury': [0],
            'diag_1_Missing': [0],
            'diag_1_Musculoskeletal': [0],
            'diag_1_Other': [0],
            'diag_1_Respiratory': [0],
            'diag_2_Diabetes': [0],
            'diag_2_Digestive': [0],
            'diag_2_Injury': [0],
            'diag_2_Missing': [0],
            'diag_2_Musculoskeletal': [0],
            'diag_2_Other': [0],
            'diag_2_Respiratory': [0],
            'diag_3_Diabetes': [0],
            'diag_3_Digestive': [0],
            'diag_3_Injury': [0],
            'diag_3_Missing': [0],
            'diag_3_Musculoskeletal': [0],
            'diag_3_Other': [0],
            'diag_3_Respiratory': [0],
            'glucose_test_no': [0],
            'glucose_test_normal': [0],
            'A1Ctest_no': [0],
            'A1Ctest_normal': [0],
            'change_yes': [0],
            'diabetes_med_yes': [0]
        }

        # Update feature_dict with actual inputs
        age = data.get('age')
        if age in feature_dict:
            feature_dict[age] = [1]

        medical_specialty = 'medical_specialty_' + data.get('medical-specialty', 'Missing')
        if medical_specialty in feature_dict:
            feature_dict[medical_specialty] = [1]

        primary_diagnosis = 'diag_1_' + data.get('primary-diagnosis', 'Missing')
        secondary_diagnosis = 'diag_2_' + data.get('secondary-diagnosis', 'Missing')
        tertiary_diagnosis = 'diag_3_' + data.get('tertiary-diagnosis', 'Missing')
        
        if primary_diagnosis in feature_dict:
            feature_dict[primary_diagnosis] = [1]
        if secondary_diagnosis in feature_dict:
            feature_dict[secondary_diagnosis] = [1]
        if tertiary_diagnosis in feature_dict:
            feature_dict[tertiary_diagnosis] = [1]

        glucose_test = 'glucose_test_' + data.get('glucose-test', 'no')
        if glucose_test in feature_dict:
            feature_dict[glucose_test] = [1]

        a1c_test = 'A1Ctest_' + data.get('a1c-test', 'no')
        if a1c_test in feature_dict:
            feature_dict[a1c_test] = [1]

        if data.get('Medication Change') == 'yes':
            feature_dict['change_yes'] = [1]
        if data.get('Diabetes Medication') == 'yes':
            feature_dict['diabetes_med_yes'] = [1]

        features = np.array(list(feature_dict.values())).reshape(1, -1)

        # Check if scaler is loaded and apply transformation
        if scaler:
            features = scaler.transform(features)
        else:
            print("Scaler not loaded. Skipping scaling.")

        # Check if models are loaded and make predictions
        # if mlp_model and rf_model:
        if rf_model:
            new_pred_proba_rf = rf_model.predict_proba(features)[:, 1]
            new_combined_proba = new_pred_proba_rf
            new_pred_combined = (new_combined_proba > 0.5).astype(int)

            # Feature Insights Graph
            feature_names = ['n_lab_procedures', 'n_medications', 'time_in_hospital', 'n_procedures']
            feature_values = [feature_dict[feature][0] for feature in feature_names]
            bar_width = 0.5
            bar_colors = ['indigo', 'firebrick', 'forestgreen', 'tomato']
            x_positions = np.arange(len(feature_names))

            fig1, ax1 = plt.subplots()
            fig1.patch.set_facecolor('none')
            ax1.set_facecolor('none')
            ax1.bar(feature_names, feature_values, width=bar_width, color=bar_colors)
            ax1.set_xlabel('Feature', labelpad=12)
            ax1.set_ylabel('Value', labelpad=15)
            ax1.set_xticks(x_positions)
            ax1.set_xticklabels(feature_names)
            # ax1.set_title('Feature Insights', pad=15)
            ax1.tick_params(axis='x', rotation=0)

            buf1 = io.BytesIO()
            plt.savefig(buf1, format='png', transparent=True)
            buf1.seek(0)
            graph_base64_feature = base64.b64encode(buf1.read()).decode('utf-8')
            plt.close(fig1)

            # Test Results Graph
            y_values = ['no', 'normal', 'high']
            color_map = {'no': 'gray', 'normal': 'green', 'high': 'red'}
            
            y_indices = []
            for key in ['glucose-test', 'a1c-test']:
                test_result = data.get(key, '').lower()
                if test_result in y_values:
                    y_indices.append(y_values.index(test_result))
                else:
                    y_indices.append(0)

            bar_colors = [color_map.get(y_values[idx], 'gray') for idx in y_indices]

            fig2, ax2 = plt.subplots()
            bars = ax2.bar(['Glucose Test', 'A1C Test'], y_indices, color=bar_colors)
            ax2.set_yticks(range(len(y_values)))
            ax2.set_yticklabels(y_values)
            ax2.set_xlabel('Test Type')
            ax2.set_ylabel('Result')
            # ax2.set_title('Test Results')

            fig2.patch.set_facecolor('none')
            ax2.set_facecolor('none')

            buf2 = io.BytesIO()
            plt.savefig(buf2, format='png', transparent=True)
            buf2.seek(0)
            graph_base64_test = base64.b64encode(buf2.read()).decode('utf-8')
            plt.close(fig2)
            # Prepare the response with predictions and graph
            response = {
                # 'mlp_prediction': mlp_model.predict(features)[0],
                'rf_prediction': rf_model.predict(features)[0],
                'combined_prediction': 'likely' if new_pred_combined[0] == 1 else 'unlikely',
                'predicted_probability': math.ceil(new_combined_proba[0] * 100),
                'risk_percentage': new_combined_proba[0] * 100,
                'feature_graph_base64':graph_base64_feature, # Add the base64 graph to the response
                'test_graph_base64': graph_base64_test
            }
        else:
            response = {
                # 'mlp_prediction': 'Model not loaded',
                'rf_prediction': 'Model not loaded',
                'combined_prediction': 'Model not loaded',
                'predicted_probability': 0,
                'risk_percentage': 0,
                'feature_graph_base64': '' , # No graph if model not loaded
                'test_graph_base64': ''    
            }
        
        return render_template('prediction.html', prediction=response)


    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return redirect(url_for('about'))


@app.route('/prediction_results')
def prediction_results():
    return render_template('prediction.html')

@app.route('/modelinsight')
def modelinsight():
    return render_template('modelinsight.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True)