from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name='EM_data_analysis',  
    version='1.0.3',  
    packages=find_packages(),  
    install_requires=[
        'opencv-python',       
        'numpy',               
        'porespy',             
        'matplotlib',          
        'openpnm',             
        'scipy',              
        'pandas',              
        'dataframe-image',     
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
)

