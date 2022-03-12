import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, params):
        super(Net, self).__init__()
        
        # Crea una capa de embebdding de tamaño vocab size que se mapea a la dimensión del embedding
        self.embedding = nn.Embedding(params.vocab_size, params.embedding_dim)

        # Crea una LSTM con tamaño de input como dim. embedding y usa lstm_hidden como hidden size
        # de la LSTM. Pone batch first para que el número de secuencias sea lo primero
        self.lstm = nn.LSTM(params.embedding_dim,
                            params.lstm_hidden_dim, batch_first=True)

        # Capa lineal para predicción del tag con número de tags como salida
        self.fc = nn.Linear(params.lstm_hidden_dim, params.number_of_tags)

    def forward(self, s):
        # Forward en capa de embedding
        s = self.embedding(s)
        
        # Forward por la lstm, devuelve las características computadas y los hidden states + cell states
        s, _ = self.lstm(s)

        # Alinea la memoria del tensor para que esté contigua en memoria [1, 3, 4, 2] -> [1, 2, 3, 4] (chunk secuencial)
        s = s.contiguous()

        # Hace un flatten en la primera dimensión para dejar un "vector de tensores" [1, B, H]
        s = s.view(-1, s.shape[2])

        # Forward por la capa linear para la predicción
        s = self.fc(s)

        # Calcula el log softax de la predicción para normalizar las probabilidades 
        # (log por eficiencia, mejor entrenamiento etc)
        return F.log_softmax(s, dim=1)



def loss_fn(outputs, labels):
    # Outputs: probabilidades de las etiquetas (log-softmax)

    # Hace un flatten del ground truth
    labels = labels.view(-1)

    # Crea una máscara para las etiquetas válidas (!=-1) y convierte True, False a 1.0, 0.0
    mask = (labels >= 0).float()

    # Labels % quita los -1 para no tener problemas de memoria [-1 a 0]
    labels = labels % outputs.shape[1]

    # No. de tokens válidos (!=-1)
    num_tokens = int(torch.sum(mask))

    # Devuelve el loss para todo el batch (entropía cruzada por palabra)
    return -torch.sum(outputs[range(outputs.shape[0]), labels]*mask)/num_tokens


def accuracy(outputs, labels):

    # Aplana el vector de labels
    labels = labels.ravel()

    # Hace un masking para las que sean !=-1
    mask = (labels >= 0)

    # Coge el argmax tras la softmax (outputs)
    outputs = np.argmax(outputs, axis=1)

    # Devuelve el accuracy
    return np.sum(outputs == labels)/float(np.sum(mask))


metrics = {
    'accuracy': accuracy,
}
