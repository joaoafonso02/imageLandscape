import os
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

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

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255.,
    horizontal_flip=True,
    shear_range=0.2,
    zoom_range=0.2,
    validation_split=0.1
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=64,
    class_mode='sparse',
    seed=2209,
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=64,
    class_mode='sparse',
    seed=2209,
    subset='validation'
)

test_datagen = ImageDataGenerator(rescale=1./255.)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='sparse',
    shuffle=False,  
    seed=2209
)

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

# Open log file
log_filename = "model_results_log.txt"
with open(log_filename, 'w') as log_file:
    log_file.write("Model Evaluation Results\n")
    log_file.write("=======================\n\n")
    
    results = {}
    for num_layers in range(1, 6):
        print(f"\n{'='*20} Training model with {num_layers} convolutional layers {'='*20}")
        log_file.write(f"\nTraining model with {num_layers} convolutional layers...\n")
        
        # Build model
        model = Sequential()
        for i in range(num_layers):
            if i == 0:
                # Assign input_shape only on first layer
                model.add(Conv2D(32 * (i + 1), (3, 3), activation='relu', input_shape=(150, 150, 3)))
            else:
                model.add(Conv2D(32 * (i + 1), (3, 3), activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D((2, 2)))
        
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(train_generator.num_classes, activation='softmax'))
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        # Train
        history = model.fit(train_generator, epochs=50, validation_data=validation_generator, callbacks=[early_stopping], verbose=1)
        
        # Evaluate
        print("\nEvaluating model...")
        test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
        
        # Get predictions
        test_generator.reset()
        pred_probs = model.predict(test_generator, verbose=1)
        pred_labels = np.argmax(pred_probs, axis=1)
        true_labels = test_generator.classes
        
        assert len(pred_labels) == len(true_labels), "Mismatch between predictions and true labels."
        
        # Calculate metrics
        class_names = list(test_generator.class_indices.keys())
        metrics_dict = classification_report(
            true_labels,
            pred_labels,
            output_dict=True,
            target_names=class_names
        )
        
        # Extract metrics
        accuracy_val = metrics_dict['accuracy']
        macro_avg = metrics_dict['macro avg']
        precision_val = macro_avg['precision']
        recall_val = macro_avg['recall']
        f1_val = macro_avg['f1-score']
        mcc_val = matthews_corrcoef(true_labels, pred_labels)
        
        # Print results to terminal
        print(f"\nResults for {num_layers}-layer model:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(true_labels, pred_labels, target_names=class_names))
        print(f"MCC: {mcc_val:.4f}")
        print(f"Accuracy (from report): {accuracy_val:.4f}")
        print(f"Macro Avg Precision: {precision_val:.4f}")
        print(f"Macro Avg Recall: {recall_val:.4f}")
        print(f"Macro Avg F1 Score: {f1_val:.4f}")
        
        # Write to log file
        log_file.write(f"Test Loss: {test_loss:.4f}\n")
        log_file.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        log_file.write("Classification Report:\n")
        report_str = classification_report(true_labels, pred_labels, target_names=class_names)
        log_file.write(report_str)
        log_file.write(f"MCC: {mcc_val:.4f}\n")
        log_file.write(f"Accuracy (from report): {accuracy_val:.4f}\n")
        log_file.write(f"Macro Avg Precision: {precision_val:.4f}\n")
        log_file.write(f"Macro Avg Recall: {recall_val:.4f}\n")
        log_file.write(f"Macro Avg F1 Score: {f1_val:.4f}\n")
        log_file.write("="*60 + "\n")
        
        # Plot confusion matrix
        plot_confusion_matrix_percentage(
            true_labels,
            pred_labels,
            class_names,
            f"confusion_matrix_{num_layers}_layers.png"
        )
        
        # Store results
        results[num_layers] = {
            'model': model,
            'history': history,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'metrics_dict': metrics_dict,
            'mcc': mcc_val,
            'accuracy': accuracy_val,
            'precision': precision_val,
            'recall': recall_val,
            'f1': f1_val
        }
    
    # Find and report best model based on test_accuracy
    best_num_layers = max(results, key=lambda x: results[x]['test_accuracy'])
    best_results = results[best_num_layers]
    
    print("\n" + "="*20 + " Best Model Results " + "="*20)
    print(f"Best model has {best_num_layers} convolutional layers")
    print(f"Test Accuracy (model.evaluate): {best_results['test_accuracy']:.4f}")
    print(f"MCC: {best_results['mcc']:.4f}")
    print(f"Accuracy (from report): {best_results['accuracy']:.4f}")
    print(f"Macro Avg Precision: {best_results['precision']:.4f}")
    print(f"Macro Avg Recall: {best_results['recall']:.4f}")
    print(f"Macro Avg F1 Score: {best_results['f1']:.4f}")
    
    # Write best model results to log file
    log_file.write("\n" + "="*20 + " Best Model Results " + "="*20 + "\n")
    log_file.write(f"Best model has {best_num_layers} convolutional layers\n")
    log_file.write(f"Test Accuracy (model.evaluate): {best_results['test_accuracy']:.4f}\n")
    log_file.write(f"MCC: {best_results['mcc']:.4f}\n")
    log_file.write(f"Accuracy (from report): {best_results['accuracy']:.4f}\n")
    log_file.write(f"Macro Avg Precision: {best_results['precision']:.4f}\n")
    log_file.write(f"Macro Avg Recall: {best_results['recall']:.4f}\n")
    log_file.write(f"Macro Avg F1 Score: {best_results['f1']:.4f}\n")
    
    # Plot learning curves for best model
    history = best_results['history']
    plt.figure(figsize=(14, 6))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
    plt.title('Accuracy over Epochs (Best Model)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
    plt.title('Loss Over Epochs (Best Model)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("best_model_learning_curves.png")
    plt.close()

print("\nExperiment completed. Results saved to log file and plots generated.")