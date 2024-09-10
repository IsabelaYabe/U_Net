import torch
import matplotlib.pyplot as plt
import numpy as np

class UNetTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, rgb_to_index, index_to_label, optimizer, criterion, device='cpu', lr=0.001, weight_decay=0.0001, save_dir='./model_checkpoints'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_dir = save_dir
        self.rgb_to_index = rgb_to_index
        self.index_to_label = index_to_label

        # Otimizador e função de perda
        self.optimizer = optimizer
        self.criterion = criterion

        # Mover o modelo para o dispositivo (GPU/CPU)
        self.model = self.model.to(self.device)

        # Dicionário para armazenar métricas de treinamento
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    def rgb_to_label(self, mask):
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int32)

        for color, label in self.rgb_to_index.items():
            matches = np.all(mask == np.array(color), axis=-1)
            label_mask[matches] = label

        return label_mask

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_one_epoch()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')

            if self.val_loader:
                val_loss, val_acc = self.validate()
                print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')

            # Salvar métricas no dicionário
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            if self.val_loader:
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

            # Exibir uma imagem predita a cada 5 épocas
            if epoch % 5 == 0:
                self.plot_sample_prediction()

        return self.history

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct_pixels = 0
        total_pixels = 0
        
        for images, labels in self.train_loader:
            images = images.to(self.device)
            if labels.shape[1] == 3:  # Verificar se tem 3 canais, indicando que ainda está em RGB
                labels = labels.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, C)
                labels = np.array([self.rgb_to_label(label) for label in labels])  # Converte cada rótulo
                labels = torch.tensor(labels, dtype=torch.long).to(self.device)  # Converte de volta para tensor
            else:
                labels = labels.to(self.device).long()
            # Forward pass
            outputs = self.model(images)
            # Converter probabilidade para rótulo
            _, predicted = torch.max(outputs.data, 1)
       
            loss = self.criterion(outputs, labels)

            # Backward pass e otimização
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            # Calcular acurácia por pixel
            _, predicted = torch.max(outputs.data, 1)
            correct_pixels += (predicted == labels).sum().item()
            total_pixels += labels.nelement()

        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100 * correct_pixels / total_pixels
        return avg_loss, accuracy
    
    def convert_to_rgb(self, predicted):
        # Verificar se o tensor tem o número correto de dimensões
        if predicted.ndim == 2:  # Caso seja [height, width]
            height, width = predicted.shape
            predicted_rgb = torch.zeros((height, width, 3), dtype=torch.float32)
        elif predicted.ndim == 3:  # Caso seja [batch_size, height, width]
            batch_size, height, width = predicted.shape
            predicted_rgb = torch.zeros((batch_size, height, width, 3), dtype=torch.float32)
        else:
            raise ValueError("O tensor 'predicted' tem uma forma inesperada: {}".format(predicted.shape))
        # Mapear cada índice de classe para o valor RGB correspondente
        for class_idx, rgb_values in self.index_to_label.items():
            rgb_values = [int(v) for v in rgb_values]  # Garantir que os valores sejam inteiros
            mask = (predicted == class_idx)
            # Aplicar os valores RGB para os pixels correspondentes à máscara
            if predicted.ndim == 2:
                for i in range(3):  # 3 canais: R, G, B
                    predicted_rgb[:, :, i][mask] = rgb_values[i]
            elif predicted.ndim == 3:
                for i in range(3):  # 3 canais: R, G, B
                    predicted_rgb[:, :, :, i][mask] = rgb_values[i]

        return predicted_rgb

    
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct_pixels = 0
        total_pixels = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).long()

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()

                # Calcular acurácia por pixel
                _, predicted = torch.max(outputs.data, 1)
                correct_pixels += (predicted == labels).sum().item()
                total_pixels += labels.nelement()

        avg_loss = running_loss / len(self.val_loader)
        accuracy = 100 * correct_pixels / total_pixels
        return avg_loss, accuracy

    def plot_sample_prediction(self):
        self.model.eval()
        with torch.no_grad():
            images, labels = next(iter(self.val_loader))  # Obter um lote do val_loader
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            _, predicted = torch.max(outputs, 1)
            
            # Converter predições para RGB usando o index_to_label
            predicted_rgb = self.convert_to_rgb(predicted)

            # Converter tensores para CPU para plotar
            images = images.cpu()
            labels = labels.cpu()
            predicted_rgb = predicted_rgb.cpu()

            # Mostrar uma imagem original, rótulo verdadeiro e rótulo predito
            plt.figure(figsize=(10, 6))
            for i in range(3):  # Mostra 3 exemplos
                plt.subplot(3, 3, i*3+1)
                plt.imshow(images[i].permute(1, 2, 0).numpy())  # (C, H, W) para (H, W, C)
                plt.title("Imagem Original")

                rgb_label = self.convert_to_rgb(labels[i]).permute(0,1,2).numpy()
                rgb_label_doubled = (rgb_label / 2).clip(0, 255).astype(np.uint8)
                plt.subplot(3, 3, i*3+2)
                plt.imshow(rgb_label_doubled)  # Ground truth em RGB
                plt.title("Ground Truth")

                rgb_predicted = predicted_rgb[i].permute(0, 1, 2).numpy()
                rgb_predicted_doubled = (rgb_predicted / 2).clip(0, 255).astype(np.uint8) 
                plt.subplot(3, 3, i*3+3)
                plt.imshow(rgb_predicted_doubled)  # Predição em RGB
                plt.title("Predição RGB")
            plt.show()


    def save_model(self, file_path='unet_model.pth'):
        torch.save(self.model.state_dict(), file_path)
        print(f"Modelo salvo em: {file_path}")

    def plot_metrics(self):
        # Plotar a perda e acurácia de treinamento/validação
        epochs = range(1, len(self.history['train_loss']) + 1)

        plt.figure(figsize=(12, 5))

        # Plotar perda
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history['train_loss'], 'b', label='Treino')
        if 'val_loss' in self.history:
            plt.plot(epochs, self.history['val_loss'], 'r', label='Validação')
        plt.title('Perda')
        plt.xlabel('Época')
        plt.ylabel('Perda')
        plt.legend()

        # Plotar acurácia
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history['train_acc'], 'b', label='Treino')
        if 'val_acc' in self.history:
            plt.plot(epochs, self.history['val_acc'], 'r', label='Validação')
        plt.title('Acurácia')
        plt.xlabel('Época')
        plt.ylabel('Acurácia (%)')
        plt.legend()

        plt.show()

# Exemplo de uso:
# trainer = UNetTrainer(model, train_loader, val_loader, device=device, lr=0.001)
# history = trainer.train(num_epochs=50)
# trainer.plot_metrics()
# trainer.save_model("unet_best_model.pth")
