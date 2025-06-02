from setuptools import setup

setup(
   name='taxi_driver',
   version='1.0',
   description='Collection of scripts to analyse nyc taxi data',
   author='Ioannis Toumpalidis',
   author_email='t_ejohn@hotmail.com',
   packages=['taxi_driver'],  #same as name
   install_requires=['numpy', 'pandas',"networkx"], #external packages as dependencies
)