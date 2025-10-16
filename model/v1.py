import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class AudioFeaturesDataset(Dataset):
    """Dataset personalizado para características de audio"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class RobustAudioClassifier(nn.Module):
    """
    Arquitectura robusta para clasificación de enfermedades basada en audio
    Incluye: Dropout, BatchNorm, Skip connections y regularización
    """
    
    def __init__(self, input_size: int, num_classes: int, hidden_sizes: list = [512, 256, 128]):
        super(RobustAudioClassifier, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Capa de entrada con normalización
        self.input_bn = nn.BatchNorm1d(input_size)
        
        # Arquitectura principal con skip connections
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            # Bloque principal
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        self.backbone = nn.Sequential(*layers)
        
        # Skip connection desde entrada
        self.skip_projection = nn.Linear(input_size, hidden_sizes[-1])
        
        # Capas de clasificación
        self.classifier = nn.Sequential(
            nn.Linear(hidden_sizes[-1], hidden_sizes[-1] // 2),
            nn.BatchNorm1d(hidden_sizes[-1] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_sizes[-1] // 2, num_classes)
        )
        
        # Inicialización de pesos
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicialización Xavier para mejor convergencia"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Normalización de entrada
        x_norm = self.input_bn(x)
        
        # Backbone
        features = self.backbone(x_norm)
        
        # Skip connection
        skip = self.skip_projection(x_norm)
        combined = features + skip
        
        # Clasificación
        output = self.classifier(combined)
        return output

class BabyDiseaseClassifier:
    """
    Clase principal para entrenar y evaluar el clasificador
    """
    
    def __init__(self, model_params: Dict[str, Any] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model_params = model_params or {}
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        print(f"Usando device: {self.device}")
    
    def load_data(self, train_path: str, test_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Carga los datos desde archivos parquet"""
        print("Cargando datos...")
        train_data = pd.read_parquet(train_path)
        
        test_data = None
        if test_path:
            test_data = pd.read_parquet(test_path)
        
        print(f"Datos de entrenamiento: {train_data.shape}")
        if test_data is not None:
            print(f"Datos de prueba: {test_data.shape}")
        
        return train_data, test_data
    
    def preprocess_data(self, data: pd.DataFrame, is_training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocesa los datos y extrae características y etiquetas"""
        
        # Separar características y etiquetas
        if 'clase' in data.columns:
            X = data.drop(['clase'], axis=1).values
            y = data['clase'].values
            
            if is_training:
                # Ajustar encoders en datos de entrenamiento
                X = self.scaler.fit_transform(X)
                y = self.label_encoder.fit_transform(y)
                self.num_classes = len(self.label_encoder.classes_)
                self.feature_size = X.shape[1]
                
                print(f"Clases encontradas: {self.label_encoder.classes_}")
                print(f"Número de características: {self.feature_size}")
            else:
                # Transformar usando encoders ya ajustados
                X = self.scaler.transform(X)
                y = self.label_encoder.transform(y)
        else:
            # Solo características (para predicción)
            X = data.values
            y = None
            if not is_training:
                X = self.scaler.transform(X)
        
        return X, y
    
    def create_model(self):
        """Crea el modelo con los parámetros especificados"""
        model_config = {
            'input_size': self.feature_size,
            'num_classes': self.num_classes,
            'hidden_sizes': self.model_params.get('hidden_sizes', [512, 256, 128])
        }
        
        self.model = RobustAudioClassifier(**model_config)
        self.model.to(self.device)
        
        print(f"Modelo creado con {sum(p.numel() for p in self.model.parameters())} parámetros")
    
    def train(self, train_data: pd.DataFrame, validation_split: float = 0.2, 
              epochs: int = 100, batch_size: int = 64, learning_rate: float = 0.001):
        """Entrena el modelo"""
        
        # Preprocesar datos
        X, y = self.preprocess_data(train_data, is_training=True)
        
        # Split entrenamiento/validación
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Crear datasets y dataloaders
        train_dataset = AudioFeaturesDataset(X_train, y_train)
        val_dataset = AudioFeaturesDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Crear modelo
        self.create_model()
        
        # Configurar optimizador y función de pérdida
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
        criterion = nn.CrossEntropyLoss()
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15
        
        print("Iniciando entrenamiento...")
        
        for epoch in range(epochs):
            # Entrenamiento
            self.model.train()
            train_loss = 0.0
            
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                
                # Gradient clipping para estabilidad
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # Validación
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
            
            # Métricas del epoch
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_acc = 100 * correct / total
            
            # Guardar historial
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Scheduler step
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Guardar mejor modelo
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'scaler': self.scaler,
                    'label_encoder': self.label_encoder,
                    'model_params': self.model_params,
                    'feature_size': self.feature_size,
                    'num_classes': self.num_classes
                }, 'best_model.pth')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}/{epochs}:')
                print(f'  Train Loss: {train_loss:.4f}')
                print(f'  Val Loss: {val_loss:.4f}')
                print(f'  Val Acc: {val_acc:.2f}%')
                print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            if patience_counter >= patience:
                print(f"Early stopping en epoch {epoch}")
                break
        
        # Cargar mejor modelo
        checkpoint = torch.load('best_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("Entrenamiento completado!")
    
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Evalúa el modelo en datos de prueba"""
        
        if self.model is None:
            raise ValueError("Modelo no entrenado. Ejecuta train() primero.")
        
        print("Evaluando modelo...")
        
        # Preprocesar datos de prueba
        X_test, y_test = self.preprocess_data(test_data, is_training=False)
        
        # Crear dataset y dataloader
        test_dataset = AudioFeaturesDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Evaluación
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_features)
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Convertir predicciones a etiquetas originales
        pred_labels = self.label_encoder.inverse_transform(all_predictions)
        true_labels = self.label_encoder.inverse_transform(all_labels)
        
        # Calcular métricas
        accuracy = accuracy_score(true_labels, pred_labels)
        
        # Reporte de clasificación
        report = classification_report(true_labels, pred_labels, output_dict=True)
        
        # Matriz de confusión
        cm = confusion_matrix(true_labels, pred_labels)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': pred_labels,
            'true_labels': true_labels,
            'probabilities': np.array(all_probabilities),
            'class_names': self.label_encoder.classes_
        }
        
        print(f"Accuracy en conjunto de prueba: {accuracy:.4f}")
        print("\nReporte de clasificación:")
        print(classification_report(true_labels, pred_labels))
        
        return results
    
    def plot_training_history(self):
        """Grafica el historial de entrenamiento"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_title('Pérdida durante el entrenamiento')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(self.history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Precisión durante el entrenamiento')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, results: Dict[str, Any]):
        """Grafica la matriz de confusión"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(results['confusion_matrix'], 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=results['class_names'],
                   yticklabels=results['class_names'])
        plt.title('Matriz de Confusión')
        plt.ylabel('Etiquetas Verdaderas')
        plt.xlabel('Predicciones')
        plt.show()
    
    def predict(self, new_data: pd.DataFrame) -> np.ndarray:
        """Realiza predicciones en nuevos datos"""
        
        if self.model is None:
            raise ValueError("Modelo no entrenado.")
        
        # Preprocesar datos
        X = self.scaler.transform(new_data.values)
        
        # Convertir a tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        # Convertir a etiquetas originales
        pred_labels = self.label_encoder.inverse_transform(predicted.cpu().numpy())
        
        return pred_labels, probabilities.cpu().numpy()
    
    def save_model(self, path: str):
        """Guarda el modelo completo"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'model_params': self.model_params,
            'feature_size': self.feature_size,
            'num_classes': self.num_classes,
            'history': self.history
        }, path)
        print(f"Modelo guardado en: {path}")
    
    def load_model(self, path: str):
        """Carga un modelo previamente entrenado"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.scaler = checkpoint['scaler']
        self.label_encoder = checkpoint['label_encoder']
        self.model_params = checkpoint['model_params']
        self.feature_size = checkpoint['feature_size']
        self.num_classes = checkpoint['num_classes']
        self.history = checkpoint.get('history', {'train_loss': [], 'val_loss': [], 'val_acc': []})
        
        self.create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Modelo cargado desde: {path}")

# Ejemplo de uso
def main():
    """Función principal de ejemplo"""
    
    # Configuración del modelo
    model_config = {
        'hidden_sizes': [512, 256, 128]  # Puedes ajustar estas capas
    }
    
    # Crear clasificador
    classifier = BabyDiseaseClassifier(model_params=model_config)
    
    # Cargar datos (ajusta las rutas según tus archivos)
    ruta_train = r"C:\Users\japal\Documents\TechCapital\Paralinguistic-Speech-Classification-for-Human-Vocalizations-2\model\audio_features_train.parquet"
    ruta_testing = r"C:\Users\japal\Documents\TechCapital\Paralinguistic-Speech-Classification-for-Human-Vocalizations-2\model\audio_features_test.parquet"
    train_data, test_data = classifier.load_data(ruta_train, ruta_testing)
    
    # Entrenar modelo
    classifier.train(
        train_data=train_data,
        validation_split=0.2,
        epochs=100,
        batch_size=64,
        learning_rate=0.001
    )
    
    # Graficar historial de entrenamiento
    classifier.plot_training_history()
    
    # Evaluar en datos de prueba
    if test_data is not None:
        results = classifier.evaluate(test_data)
        classifier.plot_confusion_matrix(results)
    
    # Guardar modelo
    classifier.save_model('baby_disease_classifier.pth')
    
    print("Proceso completado!")

if __name__ == "__main__":
    main()