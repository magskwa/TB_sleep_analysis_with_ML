
import breeds as breeds # Import the breeds module. 
import numpy as np 
import importlib
importlib.reload(breeds)

def selectBreed( breedNb: int):
    """
    Select a breed from the list of available breeds.
    :param breedNb: The breed number to select.
    :return: The list of mice names for the selected breed.
    """
    return breeds.breeds.get((breeds.breeds.keys())[breedNb])

def selectAllBreedsOfSizeN( size: int):
    """
    Select all breeds of size N.
    :param size: The size of the breeds to select.
    :return: The list of breeds name of size N.
    """
    return [breed for breed in breeds.breeds.keys() if len(breeds.breeds.get(breed)) == size]

def selectAllBreedsOfSizeNOrMore( size: int):
    """
    Select all breeds of size N or more.
    :param size: The size of the breeds to select.
    :return: The list of breeds name of size N or more.
    """
    return [breed for breed in breeds.breeds.keys() if len(breeds.breeds.get(breed)) >= size]

def getBreedName( breedNb: int) -> str :
    """
    Get the name of a breed from its number.
    :param breedNb: The breed number.
    :return: The name of the breed.
    """
    return list(breeds.breeds.keys())[breedNb]

def getMiceInBreed( breedName: str):
    """
    Get the mice names in a breed.
    :param breedName: The breed name.
    :return: The list of mice names in the breed.
    """
    return breeds.breeds.get(breedName)

def getAllmiceInAllBreeds():
    """
    Get the mice names in all breeds.
    :return: The list of mice names in all breeds.
    """
    return [mouse for breed in breeds.breeds.keys() for mouse in breeds.breeds.get(breed)]

def getFilesOfBreed( breedName: str):
    """
    Get the files of a breed.
    :param breedName: The breed name.
    :return: The list of files of the breed.
    """
    return [f"{mouse}.csv" for mouse in breeds.breeds.get(breedName)]

def getFilesOfAllBreeds():
    """
    Get the files of all breeds.
    :return: The list of files of all breeds.
    """
    return [f"{mouse}" for breed in breeds.breeds.keys() for mouse in breeds.breeds.get(breed)]

def getBreedOfMouse( mouseName: str):
    """
    Get the breed of a mouse.
    :param mouseName: The mouse name.
    :return: The breed of the mouse.
    """
    return [breed for breed in breeds.breeds.keys() if mouseName in breeds.breeds.get(breed)][0]

def getBreedIndex( breedName: str):
    """
    Get the index of a breed.
    :param breedName: The breed name.
    :return: The index of the breed.
    """
    return list(breeds.breeds.keys()).index(breedName)

def getBreedSize( breedName: str):
    """
    Get the size of a breed.
    :param breedName: The breed name.
    :return: The size of the breed.
    """
    return len(breeds.breeds.get(breedName))