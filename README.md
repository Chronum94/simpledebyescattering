# Simple Debye Scattering

This package is a minimal implementation of the Debye scattering equation. An example is below:

```python
import simpledebyescattering
from ase.cluster import wulff_construction


surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
esurf = [1.0, 1.1, 0.9]   # Surface energies.
lc = 3.61000
size = 1000  # Number of atoms
atoms = wulff_construction('Cu', surfaces, esurf,
                            size, 'fcc',
                            rounding='above', latticeconstant=lc)

# Define a scattering problem with the atoms object, the x-ray wavelength in angstrom,
# and whether to use the histogram approximation (~100x speedup with negligible loss
# of accuracy for all tests I've run.
xrd = simpledebyescattering.XrayDebye(atoms, 1.1, histogram_approximation=True)

x = np.linspace(15, 75, 1000) # 2theta angle in degrees
pattern = xrd.calc_pattern(x)
plt.plot(x, pattern)
plt.show()
```

# WARNING:

This is still a very early version. The API may change many times.
