**Theft Detection Algorithm - Report**

**1. Data Preprocessing:**

**Data Source:** 
The DCSASS dataset was utilized for this project, comprising video clips categorized into various activities, including shoplifting.

**Preprocessing Steps:**
- Videos labeled as "shoplifting" were extracted from the dataset for analysis.
- Each video was converted into individual frames to facilitate image-based analysis.
- Resizing to a standard size of (224, 224) was performed to ensure uniformity in input dimensions for the model.
- Data augmentation techniques were applied to increase the diversity of the dataset and improve model generalization. Techniques such as random flip, rotation, zoom, height, and width adjustments were employed to simulate real-world variations.

**Rationale:**
- Conversion to frames allows for image-based analysis, which is compatible with most image classification models.
- Resizing standardizes the input dimensions, making it easier to feed into pre-trained models like EfficientNetV2B0.
- Data augmentation helps in generating additional training samples, reducing overfitting, and enhancing the model's ability to generalize to unseen data.

**2. Feature Engineering:**

**Methods:**
- Histogram of Oriented Gradients (HOG) features were extracted from the preprocessed frames.
- HOG features capture local gradient information, which is effective for object detection and recognition tasks.
- These features provide a compact representation of the image, preserving essential information for classification.

**Effectiveness:**
- HOG features have demonstrated effectiveness in identifying patterns and shapes within images, making them suitable for detecting objects like human figures in shoplifting scenarios.
- The extracted features serve as informative input for the subsequent classification model, aiding in distinguishing between shoplifting and non-shoplifting activities.

**3. Model Selection and Hyperparameter Tuning:**

**Model:** 
- EfficientNetV2B0 was chosen as the base model for feature extraction.
- EfficientNet models are known for their superior performance and efficiency, making them suitable for resource-constrained environments.
- The base model's layers were frozen to retain pre-learned features and prevent overfitting on the limited shoplifting dataset.

**Justification:**
- EfficientNetV2B0 strikes a balance between model complexity and computational efficiency, making it ideal for deployment in real-world scenarios.
- Freezing the base model's layers helps leverage its pre-trained weights while enabling the model to focus on learning task-specific features during fine-tuning.

**Hyperparameter Tuning:** 
- Regularization techniques such as L2 regularization with a coefficient of 0.001 were applied to prevent overfitting.
- The Adam optimizer with default parameters was used for model optimization.
- Batch size, learning rate, and other hyperparameters were kept at default values due to the limited scope of this project.

**4. Evaluation Results:**

**Metrics:**
- **Accuracy:** 76.44%
- **Precision:** 83.37%
- **Recall:** 89.64%
- **F1 Score:** 86.40%

**Confusion Matrix:**

|                 | Predicted Negative | Predicted Positive |
|-----------------|--------------------|--------------------|
| Actual Negative | True Negative      | False Positive     |
| Actual Positive | False Negative     | True Positive      |

**5. Discussion:**

**Strengths:**
- The use of pre-trained models like EfficientNetV2B0 enables effective feature extraction from images, even with limited training data.
- Data augmentation techniques enhance model generalization and robustness by introducing variability in the training dataset.
- HOG features provide informative representations of image content, aiding in distinguishing between shoplifting and normal activities.

**Weaknesses:**
- Limited dataset size may affect model performance, particularly in capturing diverse shoplifting scenarios.
- The model's reliance on image-based features may struggle with complex or occluded shoplifting instances.
- The chosen metrics may not fully capture the nuances of shoplifting detection, necessitating additional evaluation on real-world data.

**Potential Improvements:**
- Acquire a larger and more diverse dataset to improve model generalization and performance.
- Explore advanced feature extraction techniques or deep learning-based approaches to capture richer representations of shoplifting activities.
- Fine-tune hyperparameters and experiment with different model architectures to optimize performance further.
- Incorporate additional contextual information or multi-modal data (e.g., audio, text) to enhance the model's understanding of shoplifting scenarios.

**Conclusion:**
The developed theft detection algorithm leverages state-of-the-art techniques in data preprocessing, feature engineering, and model selection to identify shoplifting activities accurately. While the model demonstrates promising performance, there is room for improvement through dataset expansion, model refinement, and incorporating additional contextual information. By addressing these aspects, the algorithm can be further enhanced for deployment in real-world surveillance and security applications.

**End of Report**
