
import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from tqdm import trange

import utils
import model.net as net
from model.data_loader import DataLoader
from evaluate import evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/small',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


def train(model, optimizer, loss_fn, data_iterator, metrics, params, num_steps):
    """
    ### Función `train`
    Realiza el entrenamiento de un modelo dado durante una época, con un optimizador y una función de pérdida, iterando
    por el conjunto de entrenamiento y almacenando las métricas correspondientes. 

    #### Parámetros:
    * `model`: instancia del modelo a entrenar.
    * `optimizer`: {torch.optim} optimizador para el entrenamiento del modelo.
    * `loss_fn`: función de pérdida a optimizar.
    * `data_iterator`: iterador que provee los datos de entrenamiento en cada step
    * `metrics`: {dict} dictionario que contiene las funciones con las métricas (accuracy, etc.)
    * `params`: {utils.Params} clase contenedora dictionario con todos los hyperparámetros del modelo, entrenamiento, etc.
    * `num_steps`: {int} número de steps para el entrenamiento en cada época 

    """
    
    # Pone el modelo en modo entrenamiento
    model.train()

    summ = []

    # Inicializa la clase de conteo para la pérdida
    loss_avg = utils.RunningAverage()

    # Itera para cada paso (step)
    t = trange(num_steps)
    for i in t:
        # Llama al iterador para que provea los siguientes datos de entrenamiento
        train_batch, labels_batch = next(data_iterator)

        # Realiza la predicción del modelo. Es hacer un forward con los datos de entrenamiento
        output_batch = model(train_batch)

        # Calcula la pérdida entre las predicciones del modelo y el ground truth (etiquetas reales)
        loss = loss_fn(output_batch, labels_batch)

        # Reseteamos los gradientes del optimizador para poder acumular de 0
        optimizer.zero_grad()

        # Hacemos el backward propagation a partir de la pérdida
        loss.backward()

        # Actualizamos los parámetros del modelo para un step
        optimizer.step()

        # Actualiza cada n steps (params.save_summary_steps)
        if i % params.save_summary_steps == 0:
            # Pasa a cpu y convierte las predicciones y el ground truth de gpu a cpu y las convierte a un array de numpy
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            # Calcula las métricas correspondientes entre las predicciones y el ground truth 
            summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                             for metric in metrics}
            # Añade el loss a las estadísticas
            summary_batch['loss'] = loss.item()

            # Actualiza el resumen del batch con las métricas y el loss
            summ.append(summary_batch)

        # Actualiza la pérdida en el conteo
        loss_avg.update(loss.item())
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()))

    # Calcula la media para todos las métricas y losses en cada paso de entrenamiento para 1 época
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}

    # Loggea las métricas en la clase Logger
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(model, train_data, val_data, optimizer, loss_fn, metrics, params, model_dir, restore_file=None):
    """
    ### Función `train_and_evaluate`
    Realiza el entrenamiento completo de un modelo, almacenando el modelo con mejor accuracy en validación mediante ModelCheckpoint.

    #### Parámetros:
    * `model`: instancia del modelo a entrenar.
    * `train_data`: conjunto de datos para entrenamiento 
    * `val_data`: conjunto de datos para validación
    * `optimizer`: {torch.optim} optimizador para el entrenamiento del modelo.
    * `loss_fn`: función de pérdida a optimizar.
    * `metrics`: {dict} dictionario que contiene las funciones con las métricas (accuracy, etc.)
    * `params`: {Params} clase contenedora dictionario con todos los hyperparámetros del modelo, entrenamiento, etc.
    * `model_dir`: {str} ruta donde se guardan los hyperparámetros del modelo
    * `restore_file`: {int} nombre del modelo donde se almacenan los pesos del modelo, etc. 
    """

    # Se carga un modelo previo en caso de que se indique la ruta
    if restore_file is not None:
        restore_path = os.path.join(
            args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))

        # Carga del modelo previo indicado
        utils.load_checkpoint(restore_path, model, optimizer)

    # Inicializamos el accuracy en validación
    best_val_acc = 0.0

    # Se itera para todas las épocas
    for epoch in range(params.num_epochs):

        # Loggea la época actual
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # Calcula el número de steps de entrenamiento de la época en función del tamaño total de entrenamiento y el tamaño de batch
        num_steps = (params.train_size + 1) // params.batch_size
        
        # Genera el iterador para los datos de entrenamiento
        train_data_iterator = data_loader.data_iterator(
            train_data, params, shuffle=True)
        
        # Entrena el modelo durante una época
        train(model, optimizer, loss_fn, train_data_iterator,
              metrics, params, num_steps)

        # Calcula el número de steps de entrenamiento de la época en función del tamaño total de validación y el tamaño de batch
        num_steps = (params.val_size + 1) // params.batch_size

        # Genera el iterador para los datos de validación
        val_data_iterator = data_loader.data_iterator(
            val_data, params, shuffle=False)
        
        # Calcula las métricas con el conjunto de validación
        val_metrics = evaluate(
            model, loss_fn, val_data_iterator, metrics, params, num_steps)

        # Accede al accuracy obtenido
        val_acc = val_metrics['accuracy']

        # Almacena un flag si se ha superado el mejor accuracy en validación hasta el momento
        is_best = val_acc >= best_val_acc

        # Guarda el mejor modelo si ha mejorado el accuracy en validación
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)

        # Loggea que se ha encontrado un mejor modelo y guarda las métricas de validación en un json
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            best_json_path = os.path.join(
                model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Guarda las métricas obtenidas en validación en la última época
        last_json_path = os.path.join(
            model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)


if __name__ == '__main__':
    
    # Parsea los argumentos
    args = parser.parse_args()
    # Carga el path que contiene los hiperparámetros del modelo y carga esos hiperparámetros
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Comprueba la disponbilidad de la GPU
    params.cuda = torch.cuda.is_available()

    # Establece las semillas para la generación de números aleatorios (también en la GPU si está disponible)
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Establece el fichero de logs
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    logging.info("Loading the datasets...")

    # Crea el dataloader para cargar los datos de entrenamiento y validación
    data_loader = DataLoader(args.data_dir, params)
    data = data_loader.load_data(['train', 'val'], args.data_dir)
    train_data = data['train']
    val_data = data['val']

    # Guarda como parámetro el tamaño de los datasets
    params.train_size = train_data['size']
    params.val_size = val_data['size']

    logging.info("- done.")

    # Crea el modelo y mete el modelo en GPU según parámetros
    model = net.Net(params).cuda() if params.cuda else net.Net(params)

    # Crea la instancia del optimizador con los parámetros del modelo a optimizar y el learning rate
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # Guarda la función de pérdida
    loss_fn = net.loss_fn

    # Guarda las métricas a calcular
    metrics = net.metrics

    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))

    # Entrena y guarda el mejor modelo con ModelCheckpoint en validación
    train_and_evaluate(model, train_data, val_data, optimizer, loss_fn, metrics, params, args.model_dir,
                       args.restore_file)
