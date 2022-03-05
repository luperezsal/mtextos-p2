import json
import logging
import os
import shutil

import torch


class Params():

    def __init__(self, json_path):
        """
        ### Función `__init__`
        Almacena los parámetros contenidos en json_path sobre la instancia self.

        #### Parámetros:
        * `self`: instancia a inicializar.
        * `json_path`: {str} ruta desde donde se leerá el archivo json.
        """
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        """
        ### Función `save`
        Almacena los parámetros contenidos en la instancia sobre el archivo json_path.

        #### Parámetros:
        * `self`: instancia desde la que se leerán los parámetros.
        * `json_path`: {str} ruta sobre la que se almacenará json.
        """
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """
        ### Función `update`
        Actualiza los parámetros contenidos en la instancia a partir del archivo json_path.

        #### Parámetros:
        * `self`: instancia sobre la que se actualizarán los parámetros.
        * `json_path`: {str} ruta desde la que se leerá el archivo json.
        """
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__


class RunningAverage():

    def __init__(self):
        """
        ### Función `__init__`
        Inicializa una instancia con los valores steps y total a cero.

        #### Parámetros:
        * `self`: instancia a inicializar.
        """
        self.steps = 0
        self.total = 0

    def update(self, val):
        """
        ### Función `update`
        Incrementa el valor de la instancia total en val unidades y el número de pasos (steps) en uno.

        #### Parámetros:
        * `self`: instancia a actualizar.
        * `val`: valor que se añadirá a total.
        """
        self.total += val
        self.steps += 1

    def __call__(self):
        """
        ### Función `__call__`
        Calcula la media en función de los parámetros total y steps.

        #### Parámetros:
        * `self`: instancia sobre la que calcular la media.

        #### Return: media calculada.
        """
        return self.total / float(self.steps)


def set_logger(log_path):
    """
    ### Función `set_logger`
    Define la activación de un archivo log alojado en log_path.

    #### Parámetros:
    * `log_path`: {str} ruta donde se almacenará el archivo log.
    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        """
        Si no hay especificado ningún handler, se inicializan:
         * File Handler: almacena los mensajes de log en el fichero especificado.
         * Stream Handler: mensajes de log en tiempo real.
        """
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """
    ### Función `save_dict_to_json`
    Almacena un diccionario d, recorriendo cada una de las entradas, como archivo json en la ruta json_path especificada.

    #### Parámetros:
    * `d`: {dict} diccionario a almacenar.
    * `json_path`: {str} ruta donde se almacenará el archivo json.
    """

    with open(json_path, 'w') as f:
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    """
    ### Función `save_checkpoint`
    Almacena en un archivo checkpoint los parámetros de entrenamineto de un modelo,
    el modelo en sí y si el modelo actual es el mejor de los que se han evaluado hasta ahora.

    #### Parámetros:
    * `state`: {dict} diccionario donde se encuentran los parámetros del modelo.
    * `is_best`: {Boolean} si el modelo es el mejor analizado hasta ahora.
    * `checkpoint`: {str} ruta donde se almacenará el archivo.
    """

    filepath = os.path.join(checkpoint, 'last.pth.tar')
    """
    Se crea el directorio especificado si no existe y guarda los parámetros incluyendo si el
    modelo es el mejor hasta ahora.
    """
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """
    ### Función `load_checkpoint`
    Carga los parámetros de entrenamineto de un modelo en el objeto model.

    #### Parámetros:
    * `checkpoint`: {str} ruta desde donde se cargará el archivo.
    * `model`: {Model} modelo donde se almacenará la carga de parámetros.
    * `optimizer`: {Optim} objeto sobre el que reanudar la ejecución del checkpoint.

    #### Return: objeto checkpoint cargado.
    """

    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint
    