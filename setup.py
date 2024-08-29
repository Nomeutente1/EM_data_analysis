from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import subprocess

with open("README.md", "r") as fh:
    long_description = fh.read()

class PostInstallCommand(install):
    def run(self):
        try:
            subprocess.check_call([os.sys.executable, "-m", "pip", "install", "--upgrade", "dataframe-image"])
            subprocess.check_call([os.sys.executable, "-m", "pip", "install", "--upgrade", "nbconvert"])
            
            installed_packages = subprocess.check_output([os.sys.executable, "-m", "pip", "freeze"]).decode("utf-8")
            if "mistune==3.0.2" in installed_packages:
                subprocess.check_call([os.sys.executable, "-m", "pip", "uninstall", "-y", "mistune"])
            
            subprocess.check_call([os.sys.executable, "-m", "pip", "install", "mistune==2.0.4"])
        except subprocess.CalledProcessError as e:
            print(f"Error occurred during post-install: {e}")
            
        install.run(self)

setup(
    name='EM_data_analysis',  
    version='1.0.6',  
    packages=find_packages(),  
    install_requires=[
        'opencv-python',       
        'numpy',               
        'porespy',             
        'matplotlib',          
        'openpnm',             
        'scipy',              
        'pandas',              
        'moviepy',             
        'IPython',             
        'scikit-image',        
    ],
    description='A Python package for EM data analysis',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Mattia Lizzano',
    author_email='mattializzano@gmail.com',
    url='https://github.com/Nomeutente1/EM_data_analysis',  
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  
    cmdclass={
        'install': PostInstallCommand,
    },
)

