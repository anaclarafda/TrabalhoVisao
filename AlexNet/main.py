try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np  # Adicionando o numpy
except ModuleNotFoundError:
    exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128 #Aqui podemos alterar o tamanho da batch
NUM_EPOCHS = 50 #Aqui podemos alterar o numero de epocas
LEARNING_RATE = 0.001 #Aqui podemos alterar o learning rate

transform = transforms.Compose([
    transforms.Resize(32), #Redimensiona as imagens para 32x32 pixels
    transforms.ToTensor(),#Converte o formato da imagem para um tensor do PyTorch
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #Normaliza os valores dos canais RGB,  𝜇=(0.5,0.5,0.5) e σ=(0.5,0.5,0.5)
    ])

# Carregar dataset CIFAR-10, carregamento eficiente das imagens 
dataset_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataset_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

# Definição da AlexNet ajustada para CIFAR-10 com Batch Normalization e Dropout
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), #Aqui podemos alterar o dropout
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3), #Aqui podemos alterar o dropout
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Instanciar modelo, loss e otimizador com regularização L2
model = AlexNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# Treinamento
def train():
    model.train()
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")

# Avaliação geral
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Acurácia no teste: {100 * correct / total:.2f}%")

#  Avaliação por classe
def test_per_class():
    model.eval()
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            for label, pred in zip(labels, predicted):
                class_total[label.item()] += 1
                if label.item() == pred.item():
                    class_correct[label.item()] += 1

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(10):
        accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f'Acurácia da classe {classes[i]}: {accuracy:.2f}%')

# Execução do treinamento e dos testes
train()
test()
test_per_class()  #  Chama a nova função aqui

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']


def generate_confusion_matrix():
    model.eval()  # Coloca o modelo em modo de avaliação
    all_labels = []
    all_predictions = []
    
    # Desabilita o cálculo de gradientes para economizar memória e computação
    with torch.no_grad():
        for images, labels in test_loader:  # Usa o test_loader para obter as imagens de teste
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # Passa as imagens pelo modelo
            _, predicted = torch.max(outputs, 1)  # Obtém as previsões
            
            all_labels.extend(labels.cpu().numpy())  # Adiciona os rótulos reais
            all_predictions.extend(predicted.cpu().numpy())  # Adiciona as previsões feitas pelo modelo
    
    return np.array(all_labels), np.array(all_predictions)

# Gerando os rótulos reais e as previsões
y_true, y_pred = generate_confusion_matrix()

# Calculando a matriz de confusão
conf_matrix = confusion_matrix(y_true, y_pred)

# Plotando a matriz de confusão
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel('Previsões')
plt.ylabel('Rótulos reais')
plt.title('Matriz de Confusão')
plt.show()


