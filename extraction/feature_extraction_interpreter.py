import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pathlib import Path

# Load .npy files
def load_npy_files(features_path, labels_path, class_names_path):
    try:
        features = np.load(features_path)
        labels = np.load(labels_path)
        class_names = np.load(class_names_path)
        return features, labels, class_names
    except FileNotFoundError as e:
        print(f"Error: {e}. Ensure .npy files exist in features_output/")
        raise
    except Exception as e:
        print(f"Error loading .npy files: {e}")
        raise

# Statistical analysis per class
def analyze_features_per_class(features, labels, class_names):
    print("=== Feature Statistics Per Class ===")
    for class_idx, class_name in enumerate(class_names):
        class_features = features[labels == class_idx]
        if len(class_features) == 0:
            print(f"No samples for {class_name}")
            continue
        print(f"\nClass: {class_name}")
        print(f"Number of samples: {len(class_features)}")
        print(f"Mean feature value: {np.mean(class_features):.4f}")
        print(f"Std feature value: {np.std(class_features):.4f}")
        print(f"Min feature value: {np.min(class_features):.4f}")
        print(f"Max feature value: {np.max(class_features):.4f}")

# Class distribution
def plot_class_distribution(labels, class_names):
    try:
        unique, counts = np.unique(labels, return_counts=True)
        plt.figure(figsize=(10, 5))
        plt.bar(class_names[unique], counts)
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        plt.savefig('class_distribution.png', bbox_inches='tight')
        plt.close()
        print("Saved class distribution plot to class_distribution.png")
    except Exception as e:
        print(f"Error plotting class distribution: {e}")

# t-SNE visualization
def plot_tsne(features, labels, class_names):
    try:
        max_samples = 1000
        if len(features) > max_samples:
            print(f"Sampling {max_samples} features for t-SNE due to large dataset")
            indices = np.random.choice(len(features), max_samples, replace=False)
            features = features[indices]
            labels = labels[indices]

        tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
        features_2d = tsne.fit_transform(features)  # Reduce to 2D
        plt.figure(figsize=(10, 8))
        for class_idx, class_name in enumerate(class_names):
            mask = labels == class_idx
            if np.sum(mask) > 0:
                plt.scatter(features_2d[mask, 0], features_2d[mask, 1], label=class_name, alpha=0.6)
        plt.title('t-SNE Visualization of Features')
        plt.legend()
        plt.savefig('tsne_visualization.png', bbox_inches='tight')
        plt.close()
        print("Saved t-SNE visualization to tsne_visualization.png")
    except Exception as e:
        print(f"Error in t-SNE visualization: {e}")

# Feature importance via SVM
def analyze_feature_importance(features, labels, class_names):
    try:
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        clf = SVC(kernel='linear', max_iter=10000)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"\n=== SVM Classifier Performance ===")
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Feature importance (absolute coefficients for linear SVM)
        importance = np.abs(clf.coef_)
        mean_importance = np.mean(importance, axis=0)
        top_features = np.argsort(mean_importance)[-10:]  # Top 10 features
        print("\n=== Top 10 Most Important Features ===")
        for idx in top_features:
            print(f"Feature {idx}: Importance = {mean_importance[idx]:.4f}")
    except Exception as e:
        print(f"Error in feature importance analysis: {e}")

# Main function
def main():
    features_path = "features_output/features.npy"
    labels_path = "features_output/labels.npy"
    class_names_path = "features_output/class_names.npy"
    
    # Load files
    try:
        features, labels, class_names = load_npy_files(features_path, labels_path, class_names_path)
    except Exception:
        return

    # Perform analyses
    analyze_features_per_class(features, labels, class_names)
    plot_class_distribution(labels, class_names)
    plot_tsne(features, labels, class_names)
    analyze_feature_importance(features, labels, class_names)

if __name__ == "__main__":
    main()
