Scene Recognition using Bag-of-Visual-Words

Overview
This project implements a Scene Recognition pipeline using the Bag-of-Visual-Words (BoVW) model with SIFT descriptors. The goal is to classify images into scene categories by extracting local descriptors, quantizing them into a visual vocabulary, and training classifiers such as SVM and k-NN.

The project includes:
SIFT feature extraction
Vocabulary construction using k-means clustering
Feature encoding (Bag-of-SIFTs histograms)
Scene classification using both Linear SVM and k-NN
Performance analysis and visualization


Repository Structure
.
├── Accuracy_vs_Vocabsize_kfold.png
├── Accuracy_vs_Vocabsize_kfold copy.png
├── Confusion_matrix.png
├── tsne_visualization.png
├── tsne_visualization_with_labels.png
├── build_vocabulary.py
├── get_bags_of_sifts.py
├── get_image_paths.py
├── get_tiny_images.py
├── nearest_neighbor_classify.py
├── proj3.py
├── svm_classify.py
├── visualize.py
├── visulizatoin.md
├── train_image_feats.pkl
├── train_labels.pkl
├── test_image_feats.pkl
├── test_labels.pkl
├── vocab.pkl


Installation
1. Clone the repository:
git clone <repo_url>
cd <repo_name>

2. Install dependencies:
pip install numpy matplotlib scikit-learn opencv-python

Usage

Step 1: Build Vocabulary
python build_vocabulary.py
Generates a vocabulary (vocab.pkl) from SIFT features extracted from training images.

Step 2: Extract Bag-of-SIFT Features
python get_bags_of_sifts.py
Encodes images into BoVW histograms.

Step 3: Train and Evaluate Classifier
python proj3.py
Runs classification with SVM or k-NN, evaluates accuracy, and produces performance plots.

Results
Accuracy vs. Vocabulary Size (k-fold) - Accuracy_vs_Vocabsize_kfold.png
Accuracy improves with vocabulary size, saturating around 800 visual words.

Confusion Matrix - Confusion_matrix.png
Most categories are well separated; confusion exists between visually similar scenes (e.g., Forest vs Mountain).


t-SNE Visualization of SIFT Descriptors - tsne_visualization.png

Sample Classification Results

See visulizatoin.md for true positives, false positives, and false negatives for each category.



Key Scripts
build_vocabulary.py – Extracts SIFT features and runs k-means to create a vocabulary.
get_bags_of_sifts.py – Encodes images into BoVW histograms.
svm_classify.py – Classifies images using Linear SVM.
nearest_neighbor_classify.py – Classifies images using k-NN.
visualize.py – Generates markdown visualization of classification results.
proj3.py – Main pipeline integrating all steps.


Conclusion
This BoVW-based system achieves competitive classification accuracy across multiple scene categories, demonstrating that local feature quantization + discriminative classifiers can effectively solve scene recognition tasks.