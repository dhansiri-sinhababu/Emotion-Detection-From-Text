# Emotion Detection from Text  
A machine learning project focused on classifying emotions from text using deep learning and NLP techniques.



## 📌 Objective  
To build an AI model capable of detecting emotions such as joy, sadness, anger, fear, and more from input text using the power of NLP and transformer-based architectures.



## 🗃️ Dataset  
- **Name**: Emotions Dataset for NLP  
- **Source**: [Kaggle - praveengovi/emotions-dataset-for-nlp](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)  
- **Format**: Semicolon-separated `.txt` files with `text;emotion`  
- **Classes**: joy, sadness, anger, fear, love, surprise


## 🛠️ Tools and Libraries  
- Python  
- **Transformers** (Hugging Face)  
- **Datasets** (Hugging Face)  
- **PyTorch**  
- **scikit-learn**  
- **matplotlib**  
- **pandas**  
- **nltk**  


## 🧠 ML Workflow  
- Load the `emotion` dataset using `datasets`  
- Text preprocessing using `nltk`  
- Tokenization using `DistilBERT tokenizer`  
- Fine-tune **DistilBERT** model for emotion classification  
- Evaluate using classification report and confusion matrix  
- Visualize emotion prediction performance  


## ✅ Results  
- Achieved **high accuracy** on multi-class emotion classification  
- Successful fine-tuning of transformer models on small labeled dataset  
- Supports real-time emotion prediction from text input  


## 📦 Files Included  
- `Emotion-Detection-From-Text.ipynb` – Main notebook with preprocessing, training, and evaluation  
- (Optional for future) `emotion_model.pt` – Trained model weights  
- (Optional for future) `tokenizer_config.json` – Tokenizer and config files if exporting model  


## 🖥️ Run Instructions  
1. Open the Jupyter Notebook  
2. Ensure internet access (required to download the model and dataset)  
3. Run all cells sequentially  
4. Enter custom text at the end to test emotion prediction  


## 🧪 Sample Prediction  
```python
text = "I can't stop smiling today, I feel amazing!"
predict_emotion(text)  # Output: joy
```


## 👨‍💻 Author  
**Name**: Dhansiri Sinhababu
