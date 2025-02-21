### Fashion-MNIST Classification using PyTorch

![PyTorch](https://img.shields.io/badge/PyTorch-1.12-red.svg)
## 📌 Overview  
This repository contains the implementation of **Fashion-MNIST classification** using **PyTorch**. The dataset consists of **70,000 grayscale images** categorized into **10 different classes**.  
Two approaches are explored in this repository:
1. **CNN-based model** for classification (Implemented in `Fashion_MNIST_Pytorch.ipynb`)
2. **Transfer Learning** using pre-trained Vgg16 model (Implemented in `TRANSFER_LEARNING_Fashion_mnist_pytorch.ipynb`)

## 📂 Dataset  
Fashion-MNIST is a dataset consisting of Zalando's article images. It contains a training set of **60,000** images and a test set of **10,000** images, each of **28×28 pixels**.  

### **Class Labels**
| Label | Class Name       |
|-------|----------------|
| 0     | T-shirt/top    |
| 1     | Trouser        |
| 2     | Pullover       |
| 3     | Dress          |
| 4     | Coat           |
| 5     | Sandal         |
| 6     | Shirt          |
| 7     | Sneaker        |
| 8     | Bag            |
| 9     | Ankle boot     |

## 🔧 Installation & Dependencies  
To set up the project locally, follow these steps:

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/Anshul21107/Fashion-mnist-Pytorch.git
cd Fashion-mnist-Pytorch
```

### **2️⃣ Install Required Packages**  
Ensure you have Python 3.7+ installed, then install dependencies using:
```bash
pip install torch torchvision matplotlib numpy
```

## 🚀 Training & Running the Model  
### **1️⃣ CNN Model Training**
Run the Jupyter Notebook `Fashion_MNIST_Pytorch.ipynb` to:
- Load the dataset  
- Train the CNN model  
- Evaluate the performance  

### **2️⃣ Transfer Learning**
Run `TRANSFER_LEARNING_Fashion_mnist_pytorch.ipynb` to:
- Use a pre-trained model for classification  
- Fine-tune the model for better accuracy  

## 📊 Model Architecture  
### **CNN Model**
The CNN architecture consists of:

- **Conv2D Layers**: Extract spatial features from the images.
- **ReLU Activation**: Introduces non-linearity.
- **MaxPooling**: Reduces dimensionality.
- **Fully Connected Layers**: Classifies the extracted features.

A sample architecture after fine tuning:
```
{'num_hidden_layers': 3,
 'neurons_per_layer': 104,
 'epochs': 50,
 'learning_rate': 0.0011205402135487422,
 'dropout_rate': 0.5,
 'batch_size': 128,
 'optimizer': 'Adam',
 'weight_decay': 1.621846402928179e-05}
)
```

## 📈 Results  
After training for several epochs, the models achieve good accuracy on the test set.

| Model Type          | Accuracy |
|--------------------|----------|
| CNN-based Model   | ~89%     |
| Transfer Learning | ~92%     |


## 👤 Author  
**Anshul Katiyar**  
- GitHub: [Anshul21107](https://github.com/Anshul21107)
- LinkedIn: [Anshul Katiyar](https://www.linkedin.com/in/anshul-katiyar-430b23235/)

---

Feel free to update the results and additional details based on your final trained models. Let me know if you need any modifications! 🚀
