AI Models for Early Cardiovascular Disease Detection and Mortality Prediction

This repository hosts an AI-powered platform for early detection and mortality prediction of cardiovascular diseases. The tool leverages advanced machine learning and deep learning models to improve clinical diagnostics and patient outcomes through timely and accurate predictions.

Access the live application here: Streamlit App.

🚀 Overview

Cardiovascular diseases (CVDs) remain the leading cause of mortality worldwide. Early detection and accurate mortality prediction can significantly reduce the burden on healthcare systems and save lives.

This application integrates AI models designed for:
	1.	Early Disease Detection: Identifies early-stage cardiovascular anomalies using patient data.
	2.	Mortality Prediction: Estimates mortality risk based on multi-modal clinical and imaging data.

🔍 Key Features
	1.	Interactive User Interface: Easy-to-use Streamlit dashboard for real-time analysis.
	2.	AI Models:
	•	Transformer-based models for multi-modal data.
	•	Machine Learning algorithms like Gradient Boosting and XGBoost.
	•	Deep Learning frameworks (CNN, ResNet) for imaging data analysis.
	3.	Multi-Modal Data Integration: Combines clinical data, EHR records, and imaging datasets for robust predictions.
	4.	Explainability: Model transparency using SHAP, Grad-CAM, and Integrated Gradients to assist clinicians in decision-making.

🏥 Clinical Applications
	•	Risk Stratification: Categorize patients based on the likelihood of developing cardiovascular diseases.
	•	Early Intervention: Aid healthcare providers in identifying high-risk patients.
	•	Mortality Analysis: Accurately Predict long-term mortality risk to improve follow-up care plans.

📊 Data Sources
	•	EHR Data: Clinical indicators and demographic details.
	•	Medical Imaging: Heart MRI/X-ray datasets.
	•	External Datasets: UK Biobank and CPRD datasets integrated for validation and testing.

⚙️ How to Use
	1.	Visit the Streamlit App.
	2.	Upload patient or imaging data.
	3.	View predictions:
	•	Disease risk scores.
	•	Mortality risk analysis.
	4.	Explore model explainability visualizations for better insights.

📈 Technology Stack
	•	Framework: Streamlit for front-end deployment.
	•	AI Models:
	•	Transformer-based architectures.
	•	CNNs for imaging.
	•	ML models like XGBoost, Random Forest.
	•	Libraries: PyTorch, TensorFlow, Scikit-learn, SHAP, OpenCV.
	•	Deployment: Streamlit Cloud.

👥 Contributors
	•	Abu Sufian
Expertise in AI for healthcare, cardiovascular imaging, and disease prediction.

🌐 Future Enhancements
	•	Real-time clinical validation.
	•	Integration with wearable device data.
	•	Addition of survival analysis models.

📐 Methodology Pipeline Diagram

A publication-quality technical methodology pipeline diagram is included in the `figures/` directory, illustrating the complete 8-stage implementation workflow from multi-modal cardiac MRI acquisition through to validated clinical decision support.

**File:** [`figures/methodology_pipeline.svg`](figures/methodology_pipeline.svg)
**Viewer:** [`figures/methodology_pipeline_viewer.html`](figures/methodology_pipeline_viewer.html)

### Pipeline Stages (Left-to-Right Flow)

| Stage | Name | Key Technical Content |
|-------|------|-----------------------|
| ① | Data Input & Cohort Assembly | T1/T2/LGE/Cine MRI · UK Biobank · Clinical metadata · `𝒳 = {X_T1, X_T2, X_LGE, X_cine, X_clinical}` |
| ② | Preprocessing & Standardisation | QC · Normalisation · Registration · Motion correction · Myocardial segmentation |
| ③ | Multi-Branch Feature Extraction | CNN · ViT · Radiomic · Temporal motion branches · `h = [h_cnn ‖ h_vit ‖ h_rad ‖ h_mot]` |
| ④ | Multi-Modal Fusion & Embedding | Cross-modal attention · Shared latent space · `z = 𝓕(h_T1, h_T2, h_LGE, h_cine, h_clin)` |
| ⑤ | Dual-Head Predictive Modelling | Classification head (abnormality) + Risk regression head (HF score) · Joint loss `ℒ = λ₁ℒ_cls + λ₂ℒ_risk` |
| ⑥ | Explainability & Biomarker Discovery | SHAP · Grad-CAM · Attention heatmaps · Regional myocardial attribution maps |
| ⑦ | Validation & Robustness Assessment | 5-fold CV · External cohort · AUC · F1 · Sensitivity · Specificity · Calibration |
| ⑧ | Clinical Output & Decision Support | Risk score dashboard · Abnormality heatmap · Risk stratification · Clinical alert system |

The diagram also includes a highlighted **Novelty and Original Contribution** block summarising the five key methodological innovations of this research.

> Diagram dimensions: 1620 × 900 px · Self-contained SVG (no external dependencies) · Suitable for academic publication, PhD thesis, and conference papers.

📝 License

This project is licensed under the MIT License.

📧 Contact

If you have any questions or collaboration opportunities, please reach out to m.sufian@uel.ac.uk

