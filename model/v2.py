import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configurar dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

class AudioDataset(Dataset):
    """Dataset personalizado para características de audio"""
    
    def __init__(self, features, labels=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels) if labels is not None else None
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]

class PositionalEncoding(nn.Module):
    """Codificación posicional para el transformer"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class CNNFeatureExtractor(nn.Module):
    """Extractor de características CNN"""
    
    def __init__(self, input_dim, cnn_channels):
        super(CNNFeatureExtractor, self).__init__()
        
        self.conv1d_layers = nn.Sequential(
            nn.Conv1d(1, cnn_channels//4, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels//4),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(cnn_channels//4, cnn_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(cnn_channels//2, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.AdaptiveAvgPool1d(1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, features)
        x = x.unsqueeze(1)  # (batch_size, 1, features)
        x = self.conv1d_layers(x)
        x = x.squeeze(-1)  # (batch_size, cnn_channels)
        return x

class LSTMProcessor(nn.Module):
    """Procesador LSTM para secuencias temporales"""
    
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(LSTMProcessor, self).__init__()
        
        # Configuración más robusta para diferentes tamaños de entrada
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Definir secuencia fija basada en características comunes de audio
        self.seq_len = 10  # Secuencia fija
        self.feature_per_step = max(1, input_dim // self.seq_len)
        
        self.lstm = nn.LSTM(
            self.feature_per_step, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.output_dim = hidden_dim * 2  # bidirectional
        
        # Capa de proyección para manejar dimensiones restantes
        remaining_features = input_dim - (self.seq_len * self.feature_per_step)
        if remaining_features > 0:
            self.projection = nn.Linear(remaining_features, self.feature_per_step)
        else:
            self.projection = None
        
    def forward(self, x):
        # x shape: (batch_size, features)
        batch_size = x.size(0)
        
        # Dividir características en secuencias
        main_features = x[:, :self.seq_len * self.feature_per_step]
        x_reshaped = main_features.view(batch_size, self.seq_len, self.feature_per_step)
        
        # Manejar características restantes si las hay
        if self.projection is not None:
            remaining_features = x[:, self.seq_len * self.feature_per_step:]
            if remaining_features.size(1) > 0:
                projected_features = self.projection(remaining_features)
                # Añadir como un paso adicional en la secuencia
                projected_features = projected_features.unsqueeze(1)  # (batch_size, 1, feature_per_step)
                x_reshaped = torch.cat([x_reshaped, projected_features], dim=1)
        
        lstm_out, (hidden, _) = self.lstm(x_reshaped)
        
        # Usar el último estado oculto (concatenar direcciones forward y backward)
        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        return final_hidden

class TransformerEncoder(nn.Module):
    """Encoder Transformer para atención global"""
    
    def __init__(self, d_model, num_heads, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        self.pos_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,  # Reducido para evitar problemas de memoria
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Proyección de entrada para ajustar dimensiones
        self.input_projection = None
        
    def forward(self, x):
        # x shape: (batch_size, features)
        batch_size, total_features = x.size()
        
        # Calcular dimensiones apropiadas
        seq_len = max(1, total_features // self.d_model)
        if seq_len == 0:
            seq_len = 1
            
        # Ajustar número de características para que sea divisible
        features_to_use = seq_len * self.d_model
        
        if features_to_use > total_features:
            # Si necesitamos más características de las disponibles, usar padding
            padding_needed = features_to_use - total_features
            x_padded = torch.cat([x, torch.zeros(batch_size, padding_needed, device=x.device)], dim=1)
            x_reshaped = x_padded.view(batch_size, seq_len, self.d_model)
        else:
            # Usar solo las características necesarias
            x_reshaped = x[:, :features_to_use].view(batch_size, seq_len, self.d_model)
        
        # Aplicar codificación posicional
        x_pos = self.pos_encoding(x_reshaped.transpose(0, 1)).transpose(0, 1)
        
        # Aplicar transformer
        transformer_out = self.transformer(x_pos)
        
        # Global average pooling
        pooled = self.global_pool(transformer_out.transpose(1, 2)).squeeze(-1)
        
        return pooled

class HybridModel(nn.Module):
    """Modelo híbrido CNN + LSTM + Transformer"""
    
    def __init__(self, input_dim, num_classes, model_params):
        super(HybridModel, self).__init__()
        
        self.input_dim = input_dim
        
        # Extractores de características
        self.cnn_extractor = CNNFeatureExtractor(
            input_dim, 
            model_params['cnn_channels']
        )
        
        self.lstm_processor = LSTMProcessor(
            input_dim,
            model_params['lstm_hidden']
        )
        
        self.transformer_encoder = TransformerEncoder(
            model_params['transformer_dim'],
            model_params['num_heads'],
            model_params['num_transformer_layers']
        )
        
        # Capa de fusión - calcular dimensión correcta
        fusion_dim = (model_params['cnn_channels'] + 
                     self.lstm_processor.output_dim + 
                     model_params['transformer_dim'])
        
        print(f"Dimensiones de fusión: CNN={model_params['cnn_channels']}, "
              f"LSTM={self.lstm_processor.output_dim}, "
              f"Transformer={model_params['transformer_dim']}, "
              f"Total={fusion_dim}")
        
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Verificar dimensiones de entrada
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input, got {x.dim()}D")
        
        # Extraer características con cada componente
        cnn_features = self.cnn_extractor(x)
        lstm_features = self.lstm_processor(x)
        transformer_features = self.transformer_encoder(x)
        
        # Verificar dimensiones antes de concatenar
        print(f"CNN features shape: {cnn_features.shape}")
        print(f"LSTM features shape: {lstm_features.shape}")
        print(f"Transformer features shape: {transformer_features.shape}")
        
        # Fusionar características
        combined_features = torch.cat([
            cnn_features, 
            lstm_features, 
            transformer_features
        ], dim=1)
        
        # Clasificación final
        output = self.fusion_layers(combined_features)
        
        return output

class HybridBabyDiseaseClassifier:
    """Clasificador híbrido para enfermedades de bebés"""
    
    def __init__(self, model_params):
        self.model_params = model_params
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.feature_columns = None
        
    def load_data(self, train_path, test_path):
        """Cargar datos de entrenamiento y prueba"""
        print("Cargando datos...")
        
        # Cargar datos
        train_data = pd.read_parquet(train_path)
        test_data = pd.read_parquet(test_path)
        
        print(f"Datos de entrenamiento: {train_data.shape}")
        print(f"Datos de prueba: {test_data.shape}")
        print(f"Clases en entrenamiento: {train_data['clase'].unique()}")
        
        return train_data, test_data
    
    def preprocess_data(self, train_data, test_data):
        """Preprocesar los datos"""
        print("Preprocesando datos...")
        
        # Identificar columnas de características
        exclude_cols = ['nombre_archivo', 'clase']
        self.feature_columns = [col for col in train_data.columns 
                               if col not in exclude_cols]
        
        # Extraer características y labels
        X_train = train_data[self.feature_columns].values
        y_train = train_data['clase'].values
        X_test = test_data[self.feature_columns].values
        
        # Manejar valores faltantes
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalizar características
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Codificar labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        return X_train_scaled, X_test_scaled, y_train_encoded, test_data['nombre_archivo']
    
    def train_model(self, X_train, y_train, validation_split=0.2):
        """Entrenar el modelo híbrido"""
        print("Entrenando modelo híbrido...")
        
        # Dividir en entrenamiento y validación
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=validation_split, 
            stratify=y_train, random_state=42
        )
        
        # Crear datasets
        train_dataset = AudioDataset(X_train_split, y_train_split)
        val_dataset = AudioDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Crear modelo
        input_dim = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        
        self.model = HybridModel(input_dim, num_classes, self.model_params)
        self.model.to(device)
        
        # Configurar entrenamiento
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # Entrenamiento
        best_val_acc = 0.0
        train_losses = []
        val_accuracies = []
        
        for epoch in range(100):  # Máximo 100 épocas
            # Entrenamiento
            self.model.train()
            total_loss = 0.0
            
            for batch_features, batch_labels in tqdm(train_loader, 
                                                   desc=f"Época {epoch+1}"):
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            # Validación
            val_acc = self.evaluate(val_loader)
            val_accuracies.append(val_acc)
            
            print(f"Época {epoch+1}: Loss={avg_loss:.4f}, Val_Acc={val_acc:.4f}")
            
            # Guardar mejor modelo
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_hybrid_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= 10:
                print("Early stopping activado")
                break
            
            # Ajustar learning rate
            scheduler.step(val_acc)
        
        # Cargar mejor modelo
        self.model.load_state_dict(torch.load('best_hybrid_model.pth'))
        
        # Mostrar curvas de entrenamiento
        self.plot_training_curves(train_losses, val_accuracies)
        
        return best_val_acc
    
    def evaluate(self, data_loader):
        """Evaluar el modelo"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in data_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = self.model(batch_features)
                _, predicted = torch.max(outputs.data, 1)
                
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        return correct / total
    
    def predict(self, X_test, file_names):
        """Realizar predicciones en datos de prueba"""
        print("Realizando predicciones...")
        
        self.model.eval()
        test_dataset = AudioDataset(X_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        all_predictions = []
        
        with torch.no_grad():
            for batch_features in tqdm(test_loader, desc="Prediciendo"):
                if isinstance(batch_features, tuple):
                    batch_features = batch_features[0]
                
                batch_features = batch_features.to(device)
                outputs = self.model(batch_features)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
        
        # Decodificar predicciones
        predicted_labels = self.label_encoder.inverse_transform(all_predictions)
        
        # Crear DataFrame de resultados
        results_df = pd.DataFrame({
            'nombre_archivo': file_names,
            'clase': predicted_labels
        })
        
        return results_df
    
    def plot_training_curves(self, train_losses, val_accuracies):
        """Graficar curvas de entrenamiento"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss de entrenamiento
        ax1.plot(train_losses)
        ax1.set_title('Pérdida de Entrenamiento')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Precisión de validación
        ax2.plot(val_accuracies)
        ax2.set_title('Precisión de Validación')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

def main_hybrid():
    """Función principal para el modelo híbrido"""
    
    # Configuración optimizada para dataset pequeño
    hybrid_config = {
        'cnn_channels': 128,      # Reducido para dataset pequeño
        'lstm_hidden': 64,        # Reducido para evitar overfitting
        'transformer_dim': 64,    # Reducido para compatibilidad
        'num_heads': 4,           # Reducido (debe dividir transformer_dim)
        'num_transformer_layers': 2  # Reducido para dataset pequeño
    }
    
    print("Configuración del modelo:")
    for key, value in hybrid_config.items():
        print(f"  {key}: {value}")
    
    # Crear clasificador híbrido
    classifier = HybridBabyDiseaseClassifier(model_params=hybrid_config)
    
    # Cargar datos
    ruta_train = r"C:\Users\japal\Documents\TechCapital\Paralinguistic-Speech-Classification-for-Human-Vocalizations-2\model\audio_features_train.parquet"
    ruta_testing = r"C:\Users\japal\Documents\TechCapital\Paralinguistic-Speech-Classification-for-Human-Vocalizations-2\model\audio_features_test.parquet"
    
    train_data, test_data = classifier.load_data(ruta_train, ruta_testing)
    
    # Preprocesar datos
    X_train, X_test, y_train, file_names = classifier.preprocess_data(train_data, test_data)
    
    print(f"Forma de X_train: {X_train.shape}")
    print(f"Forma de X_test: {X_test.shape}")
    print(f"Número de clases: {len(np.unique(y_train))}")
    
    # Entrenar modelo
    best_accuracy = classifier.train_model(X_train, y_train)
    print(f"Mejor precisión de validación: {best_accuracy:.4f}")
    
    # Realizar predicciones
    predictions_df = classifier.predict(X_test, file_names)
    
    # Guardar resultados
    output_path = "predicciones_enfermedades_bebes.csv"
    predictions_df.to_csv(output_path, index=False)
    print(f"Predicciones guardadas en: {output_path}")
    
    # Mostrar distribución de predicciones
    print("\nDistribución de predicciones:")
    print(predictions_df['clase'].value_counts())
    
    return classifier, predictions_df

if __name__ == "__main__":
    classifier, predictions = main_hybrid()