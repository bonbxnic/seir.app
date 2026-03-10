from setuptools import setup, find_packages

setup(
    name='seir_framework',
    version='0.1.0',
    packages=find_packages(),
    author='Your Name',
    author_email='your.email@example.com',
    description='A framework for SEIR modeling and analysis.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='http://your-repository-url.com',  # Optional
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Choose an appropriate license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
