from setuptools import find_packages,setup
from typing import List
HYPEN_E_DOT ='-e .'
def get_requirments(file_path:str)->List[str]:
    '''this function will return the list of requirments'''
    requiremets=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements: 
            requirements.remove(HYPEN_E_DOT)
    return requirements


setup(
    name='mlproject',
    version='0.0.1',
    author='Shiv',
    author_email='shivammishra01329@gmail.com',
    packages=find_packages(),
    installs_requires=get_requirments('requirements.txt')


)