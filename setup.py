from setuptools import setup, find_packages

setup(
    name='echodaft',  
    version='0.1.0',
    description='',
    author='Eda Özkaynar',  
    author_email='ozkaynar@metu.edu.tr',  
    packages=find_packages(),  # Tüm alt klasörleri dahil eder (içinde __init__.py varsa)
    install_requires=[
        'torch>=1.13',
        'torchvision',
        'numpy',
        'pandas',
        'opencv-python',
        'Pillow',
        'scikit-image',
        'SimpleITK',
        'imageio',
        'tqdm',
        'matplotlib',
        'scikit-learn',
        'click',
        'pydicom',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    entry_points={
        "console_scripts": [
            "echodaft=echodaft:main",
            ],
    },

)