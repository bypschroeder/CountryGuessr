# Country Prediction from Google Street View Images Using Machine Learning

Group 3: Patrick Schröder, Ralf Manig, Annika Pätzold

## Project Structure

- `prepare_data.ipynb`: Jupyter notebook that downloads, formats, merges, and balances datasets to create our custom dataset.
- `train_resnet.ipynb`: Jupyter notebook for training the ResNet50 model and running inference on the test set.
- `train_vit.ipynb`: Jupyter notebook for training the Vision Transformer model and running inference on the test set.
- `eval_resnet.ipynb`: Jupyter notebook that evaluates a trained ResNet50 model by computing global and per-class classification metrics, visualizing performance to highlight class-wise strengths and weaknesses, and applying Grad-CAM heatmaps on misclassified samples to interpret failure cases.
- `eval_vit.ipynb`: Jupyter notebook that evaluates a trained Vision Transformer model by computing global and per-class classification metrics, visualizing performance to highlight class-wise strengths and weaknesses, and applying Grad-CAM heatmaps on misclassified samples to interpret failure cases.
- `eval_compare.ipynb`: Jupyter notebook that directly compares trained ResNet50 and Vision Transformer models by computing overall and per-class metrics, visualizing performance differences, and identifying outliers.
- `google_test_imgs.ipynb`: Jupyter notebook that generates test images using the Google Street View Static API.
- `showcase.ipynb`: Jupyter notebook showcasing trained model performance on locations imaged via the Google Street View Static API.
- `src/`: Contains Python scripts with classes and functions for dataset loading, model setup, training, validation, and inference.
