import argparse
import logging
import os

import numpy as np
import torch
import utils
import model.net as net
from model.data_loader import DataLoader

# Define los parámetros disponibles para ejecutar y los inicializa a un valor por defecto.
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/small', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


def evaluate(model, loss_fn, data_iterator, metrics, params, num_steps):
    """
    ### Función `evaluate`
    Evalúa el modelo en el conjunto de test y devuelve las métricas correspondientes

    #### Parámetros:
    * `model`: instancia del modelo a evaluar.
    * `loss_fn`: función de pérdida a computar.
    * `data_iterator`: iterador con los datos de test
    * `metrics`: {dict} diccionario con las métricas a evaluar.
    * `params`: {dict} hyperparámetros para el test (NOT USED).
    * `num_steps`: {dict} número de pasos a dar para evaluar el modelo.

    #### Return `metrics_mean`: devuelve un diccionario con los resultados de las métricas
    """
    
    # Configura el modelo en modo evaluación, desactivando ciertas funciones como Dropout.
    model.eval()

    summ = []

    for _ in range(num_steps):

        # Obtiene un batch y sus respectivas etiquetas del conjunto de datos para el batch actual.
        data_batch, labels_batch = next(data_iterator)
   
        # Obtiene la predicción del modelo para los datos del batch.
        output_batch = model(data_batch)

        # Calcula la función de pérdida (entropía cruzada) para todo el batch respecto a sus etiquetas.
        loss = loss_fn(output_batch, labels_batch)
    
        # A partir del tensor output_batch, se traspasan los datos y las etiquetas a CPU.
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        # Calcula cada una de las métricas disponibles pasadas como parámetros, en el caso del modelo Net, "accuracy".
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}

        # Almacena la función de pérdida en el sumario del batch actual.
        summary_batch['loss'] = loss.item()

        # Añade a al conjunto de sumarios el del batch actual.
        summ.append(summary_batch)

    # Calcula la media de cada una de las métricas calculadas para cada batch.
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}

    # Transforma las métricas a formato String para poder visualizarlas.
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)

    return metrics_mean


if __name__ == '__main__':

    # Se leen los parámetros y se almacenan en sus respectivas variables.
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    params.cuda = torch.cuda.is_available()     # use GPU is available

    # Establece una semilla en CPU, si está disponible la GPU también la inicializa en ella.
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)
        
    # Inicializa el archivo log para escribir en él.
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    logging.info("Creating the dataset...")

    # Se inicializa una instancia DataLoader que cargará los datos desde el directorio especificado y los parámetros.
    data_loader = DataLoader(args.data_dir, params)

    # Se almacenan los datos del conjunto de test (directorio) una vez cargado el DataLoader.
    data = data_loader.load_data(['test'], args.data_dir)
    test_data = data['test']

    # Se obtiene el tamaño del conjunto de test y se crea un iterador para poder recorrer las muestras junto con sus etiquetas.
    params.test_size = test_data['size']
    test_data_iterator = data_loader.data_iterator(test_data, params)

    logging.info("- done.")

    # Se crea un modelo Net.
    model = net.Net(params).cuda() if params.cuda else net.Net(params)
    
    # Obtiene la función de pérdida y las métricas (accuracy) de la clase net.py
    loss_fn = net.loss_fn
    metrics = net.metrics
    
    logging.info("Starting evaluation")

    # Carga el modelo que se quiere evaluar
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Se calcula el número de iteraciones de la evaluación en función del tamaño del conjunto de test
    # y el tamaño del batch especificado.
    num_steps = (params.test_size + 1) // params.batch_size

    # Se llama a la función evaluate con los parámetros especificados anteriormente.
    test_metrics = evaluate(model, loss_fn, test_data_iterator, metrics, params, num_steps)

    # Se establece la ruta y se guardan las métricas del resultado de la evaluación del modelo.
    save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
