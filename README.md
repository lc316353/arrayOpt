# arrayOpt
A package that allows to optimize seismometer positions for Newtonian noise mitigation with efficient algorithms in different geometries and constrained volumes

For usage please refer to the example files and docstrings. Hopefully, more documentation will follow.

The code published here is used in the paper 'Optimization and robustness of cost-efficient seismic arrays for Newtonian noise cancellation at the Einstein Telescope' by Patrick Schillings and Johannes Erdmann (2026), not yet published.

It is based on code from Francesca Badaracco (https://github.com/LaBadda/Newtonian_Noise_optimizations).

It is a tool that is provided for the ET-collaboration.


## Dependencies

In a new virtual environment run

    pip install pyswarms==1.3.0
    pip install jax==0.4.30
    pip install optax==0.2.4

