from setuptools import setup, find_packages

setup(
    name="EM_data_analysis",
    version="1.0",
    description="This package contains several modules for Electron Microscopy data analysis",
    author="Mattia Lizzano",
    author_email="mattializzano@gmail.com",
    url="https://github.com/Nomeutente1/EM_data_analysis",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

