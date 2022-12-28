# Simple Debye Scattering

This package is a minimal implementation of the Debye scattering equation. An example is below:

```python
import numpy as np
import matplotlib.pyplot as plt
import simpledebyescattering
from ase.cluster import wulff_construction
from time import time as t

surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
esurf = [1.0, 1.1, 0.9]   # Surface energies.
lc = 3.61000
size = 2000  # Number of atoms
atoms = wulff_construction('Cu', surfaces, esurf,
                            size, 'fcc',
                            rounding='above', latticeconstant=lc)

# Define a scattering problem with the atoms object, the x-ray wavelength in angstrom,
# and whether to use the histogram approximation (~100x speedup with negligible loss
# of accuracy for all tests I've run.
x = np.linspace(15, 75, 500) # 2theta angle in degrees
xrd = simpledebyescattering.XrayDebye(atoms, 1.54, histogram_approximation=True)
tstart = t()
pattern = xrd.calc_pattern(x)
print(f"Time for perfect Wulff nanoparticle: {(t() - tstart):0.3f} s.")
plt.plot(x, pattern)

atoms.rattle(stdev=0.05)
xrd = simpledebyescattering.XrayDebye(atoms, 1.54, histogram_approximation=True)
tstart = t()
pattern = xrd.calc_pattern(x)
print(f"Time for rattled Wulff nanoparticle: {(t() - tstart):0.3f} s.")
plt.plot(x, pattern)
plt.show()
```

Because I like performance tests, let's do a performance plot:

```python
import numpy as np
import perfplot
import simpledebyescattering
from ase.cluster import wulff_construction


def setup_atoms(n):
    surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
    esurf = [1.0, 1.1, 0.9]   # Surface energies.
    lc = 3.61000
    size = n  # Number of atoms
    atoms = wulff_construction('Cu', surfaces, esurf,
                                size, 'fcc',
                                rounding='above', latticeconstant=lc)
    return atoms

def do_thing1(atoms):
    x = np.linspace(15, 75, 500)
    xrd = simpledebyescattering.XrayDebye(atoms, 1.54, histogram_approximation=True)
    pattern = xrd.calc_pattern(x)

def do_thing2(atoms):
    atoms.rattle(stdev=0.05)
    x = np.linspace(15, 75, 500)
    xrd = simpledebyescattering.XrayDebye(atoms, 1.54, histogram_approximation=True)
    pattern = xrd.calc_pattern(x)


perfplot.show(
    setup=lambda n: setup_atoms(n),  # or setup=np.random.rand
    kernels=[
        do_thing1,
        do_thing2,
    ],
    labels=["Pristine", "Rattled"],
    n_range=np.ceil(np.logspace(3, 4.5, 5)),
    xlabel="# of atoms",
    # More optional arguments with their default values:
    logx=True,  # set to True or False to force scaling
    logy=True,
    equality_check=None,  # set to None to disable "correctness" assertion
)
```
The above example uses ![perfplot](https://github.com/nschloe/perfplot) to make a performance plot of two simple sets of jobs.

# WARNING:

This is still a very early version. The API may change many times.
