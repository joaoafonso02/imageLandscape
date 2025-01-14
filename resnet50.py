import os
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Set random seed for reproducibility
seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Define directories
data_dir = 'archive/'
prediction_path = 'archive/seg_pred/'
train_dir = os.path.join(data_dir, 'seg_train/seg_train')
test_dir = os.path.join(data_dir, 'seg_test/seg_test')
pred_dir = prediction_path

# Ensure prediction directory exists
if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)

# Gather all image paths and labels
image_paths = []
labels = []
class_names = sorted(os.listdir(train_dir))
for class_name in class_names:
    class_folder = os.path.join(train_dir, class_name)
    if not os.path.isdir(class_folder):
        continue
    for fname in os.listdir(class_folder):
        fpath = os.path.join(class_folder, fname)
        if os.path.isfile(fpath):
            image_paths.append(fpath)
            labels.append(class_names.index(class_name))

image_paths = np.array(image_paths)
labels = np.array(labels)

# Define k-fold cross-validation
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed_value)

# Function to plot confusion matrix
def plot_confusion_matrix_percentage(true_labels, pred_labels, class_names, filename):
    cm = confusion_matrix(true_labels, pred_labels)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    annotations = np.array([[f'{value:.1f}%' for value in row] for row in cm_percentage])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percentage, annot=annotations, fmt="", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, cbar=False)
    plt.title(f"Confusion Matrix ({filename})", fontsize=16)
    plt.xlabel("Predicted Labels", fontsize=12)
    plt.ylabel("True Labels", fontsize=12)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Initialize lists to store aggregate metrics
fold_accuracies = []
fold_losses = []
fold_f1_scores = []
fold_mccs = []
aggregate_confusion = np.zeros((len(class_names), len(class_names)), dtype=int)

# Open log file
log_filename = os.path.join(pred_dir, "model_results_log.txt")
with open(log_filename, 'w') as log_file:
    log_file.write("ResNet50 K-Fold Cross-Validation Results\n")
    log_file.write("=========================================\n\n")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels), 1):
        print(f"\n{'='*20} Fold {fold}/{k} {'='*20}")
        log_file.write(f"\n{'='*20} Fold {fold}/{k} {'='*20}\n")
        
        # Split data
        X_train, X_val = image_paths[train_idx], image_paths[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        
        # Create DataFrame
        train_df = pd.DataFrame({'filename': X_train, 'class': y_train})
        val_df = pd.DataFrame({'filename': X_val, 'class': y_val})
        
        # Data Generators
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            horizontal_flip=True,
            shear_range=0.2,
            zoom_range=0.2
        )
        
        val_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )
        
        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            x_col='filename',
            y_col='class',
            target_size=(150, 150),
            batch_size=64,
            class_mode='raw',
            shuffle=True,
            seed=seed_value
        )
        
        validation_generator = val_datagen.flow_from_dataframe(
            val_df,
            x_col='filename',
            y_col='class',
            target_size=(150, 150),
            batch_size=64,
            class_mode='raw',
            shuffle=False,
            seed=seed_value
        )
        
        # Build ResNet50 Model
        print(f"\nBuilding ResNet50 Model for Fold {fold}")
        log_file.write(f"\nBuilding ResNet50 Model for Fold {fold}\n")
        
        resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
        resnet_base.trainable = False
        
        model = Sequential()
        model.add(resnet_base)
        model.add(GlobalAveragePooling2D())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(class_names), activation='softmax'))
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        # Train the model
        print(f"\nTraining ResNet50 Model for Fold {fold}")
        log_file.write(f"\nTraining ResNet50 Model for Fold {fold}\n")
        history = model.fit(
            train_generator, 
            epochs=50, 
            validation_data=validation_generator, 
            callbacks=[early_stopping], 
            verbose=1
        )
        
        # Evaluate the model
        print(f"\nEvaluating ResNet50 Model for Fold {fold}")
        log_file.write(f"\nEvaluating ResNet50 Model for Fold {fold}\n")
        val_loss, val_accuracy = model.evaluate(validation_generator, verbose=1)
        fold_losses.append(val_loss)
        fold_accuracies.append(val_accuracy)
        
        # Predictions
        validation_generator.reset()
        pred_probs = model.predict(validation_generator, verbose=1)
        pred_labels = np.argmax(pred_probs, axis=1)
        true_labels = y_val  # Alternatively: validation_generator.classes
        
        # Calculate metrics
        metrics_dict = classification_report(
            true_labels,
            pred_labels,
            output_dict=True,
            target_names=class_names,
            zero_division=0
        )
        
        macro_f1 = metrics_dict['macro avg']['f1-score']
        mcc = matthews_corrcoef(true_labels, pred_labels)
        fold_f1_scores.append(macro_f1)
        fold_mccs.append(mcc)
        
        # Aggregate confusion matrices
        aggregate_confusion += confusion_matrix(true_labels, pred_labels)
        
        # Print and log results
        print(f"\nResults for Fold {fold}:")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Macro Avg F1-Score: {macro_f1:.4f}")
        print(f"MCC: {mcc:.4f}\n")
        print(classification_report(true_labels, pred_labels, target_names=class_names, zero_division=0))
        
        # Write results to log file
        log_file.write(f"Validation Loss: {val_loss:.4f}\n")
        log_file.write(f"Validation Accuracy: {val_accuracy:.4f}\n")
        log_file.write(f"Macro Avg F1-Score: {macro_f1:.4f}\n")
        log_file.write(f"MCC: {mcc:.4f}\n")
        log_file.write("Classification Report:\n")
        report_str = classification_report(true_labels, pred_labels, target_names=class_names, zero_division=0)
        log_file.write(report_str)
        log_file.write("="*60 + "\n")
        
        # Plot and save confusion matrix for the fold
        cm_filename = f"confusion_matrix_fold_{fold}.png"
        plot_confusion_matrix_percentage(
            true_labels,
            pred_labels,
            class_names,
            cm_filename
        )
        log_file.write(f"Confusion matrix saved to {cm_filename}\n")
        
        # Save the model for this fold
        model_save_path = os.path.join(pred_dir, f"resnet50_model_fold_{fold}.h5")
        model.save(model_save_path)
        log_file.write(f"ResNet50 Model saved to {model_save_path}\n")
        
        # Plot and save learning curves for the fold
        plt.figure(figsize=(14, 6))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
        plt.title(f'Accuracy over Epochs (Fold {fold})')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss', marker='o')
        plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
        plt.title(f'Loss Over Epochs (Fold {fold})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        learning_curves_path = f"learningcurves_resnet50_fold_{fold}.png"
        plt.savefig(learning_curves_path)
        plt.close()
        log_file.write(f"Learning curves saved to {learning_curves_path}\n")

    # After all folds, calculate aggregate metrics
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    mean_loss = np.mean(fold_losses)
    std_loss = np.std(fold_losses)
    mean_f1 = np.mean(fold_f1_scores)
    std_f1 = np.std(fold_f1_scores)
    mean_mcc = np.mean(fold_mccs)
    std_mcc = np.std(fold_mccs)
    
    # Log aggregate results
    log_file.write("\nCross-Validation Aggregate Results:\n")
    log_file.write(f"Mean Validation Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}\n")
    log_file.write(f"Mean Validation Loss: {mean_loss:.4f} ± {std_loss:.4f}\n")
    log_file.write(f"Mean Macro F1-Score: {mean_f1:.4f} ± {std_f1:.4f}\n")
    log_file.write(f"Mean MCC: {mean_mcc:.4f} ± {std_mcc:.4f}\n\n")
    
    # Print aggregate results
    print("\nCross-Validation Aggregate Results:")
    print(f"Mean Validation Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Mean Validation Loss: {mean_loss:.4f} ± {std_loss:.4f}")
    print(f"Mean Macro F1-Score: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Mean MCC: {mean_mcc:.4f} ± {std_mcc:.4f}")
    
    # Plot aggregate confusion matrix
    aggregate_cm_filename = "aggregate_confusion_matrix.png"
    plot_confusion_matrix_percentage(
        aggregate_confusion.argmax(axis=1),  # Assuming you want to represent cumulative predictions
        aggregate_confusion.argmax(axis=0),
        class_names,
        aggregate_cm_filename
    )
    log_file.write(f"Aggregated confusion matrix saved to {aggregate_cm_filename}\n")
    
    # Optionally, plot aggregate learning curves or other summaries
    


#### **Visualizing Aggregate Metrics**

# Plot Mean and Std Dev for Accuracy and Loss
metrics = {
    'Accuracy': (mean_accuracy, std_accuracy),
    'Loss': (mean_loss, std_loss),
    'Macro F1-Score': (mean_f1, std_f1),
    'MCC': (mean_mcc, std_mcc)
}

labels = list(metrics.keys())
means = [metrics[label][0] for label in labels]
stds = [metrics[label][1] for label in labels]

plt.figure(figsize=(10, 6))
sns.barplot(x=labels, y=means, yerr=stds, capsize=0.2, palette='viridis')
plt.title('Mean ± Std Dev of Performance Metrics across Folds')
plt.ylabel('Score')
plt.ylim(0, 1)  # Adjust based on metric ranges
plt.tight_layout()
plt.savefig("aggregate_metrics.png")
plt.close()
log_file.write(f"Aggregate metrics bar chart saved to aggregate_metrics.png\n")