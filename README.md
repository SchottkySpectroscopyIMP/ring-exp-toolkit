# Ring Experiments Toolkit
During a heavy-ion storage-ring experiment, one is often in the need of a specific _calculator_ to estimate certain beam kinetic parameters for beam tuning or strategy planning in a timely manner.
This toolkit does such jobs.
It uses the **CSRe@IMP** as the example ring, but can easily be adapted to other storage rings by changing the machine parameters in the code to their particular values.

The code is written in `Python 3` and mainly consists of two scripts **`utility.py`** and **`ion_id.py`**.
It is preferably imported into a `IPython` session or a `Jupyter` notebook for interactive processing.

## Functionality
`utility.py`:
 - conversions between ion's velocity (β and γ), revolution frequency, magnetic rigidity, kinetic energy, as well as the peak location in the Schottky spectrum when the center frequency and span are given

`ion_id.py`:
 - estimate the location and strength of an ion signal given the [`LISE++`](http://lise.nscl.msu.edu/lise.html) simulation result
 - filter out the unqualified ion candidates owing to the low yields and/or the short half-lives
 - identify ions given a Schottky spectrum with multiple peaks

## Installation
Make sure the following requirements are fulfilled.

### Prerequisites
 - `Python 3`
 - `Numpy`, `Pandas`
 - [atomic mass database](http://amdc.in2p3.fr/masstables/Ame2016/mass16.txt) `mass16.txt`
 - [atomic half-life database](http://amdc.in2p3.fr/nubase/nubase2016.txt) `nubase2016.txt`
 - [electron binding energy]( https://physics.nist.gov/PhysRefData/ASD/ionEnergy.html) `electron_binding_energy.csv`
 - `IPython`/`Jupyter` (_optional_)
 - `LISE++` simulation files (_only if ion identification is needed_)
 - `iid_GUI.ui` ui file (_only if GUI is used_)

To run the code, launch a `IPython` session or a `Jupyter` notebook and import the scripts as packages.
Simple as that!

## Usage
Examples are
```python
import utility as _util
util = _util.Utility(242.9, 500) # frequency window specified by center frequency in MHz, and span in kHz
util.set_ion("58Ni19")
util.set_energy(143.92) # kinetic energy in MeV/u
```

```python
import ion_id as _iid
iid = _iid.IID("58Ni28.lpp", 242.9, 500) # LISE++ simulation file, center frequency in MHz, span in kHz
iid.calibrate_peak_loc("58Ni28", -140, 161) # identified ion, peak location in kHz within the frequency window, harmonic
```

```cmd
python iid_GUI.py
```

For figure-assisted tutorials, see [Wiki](https://github.com/SchottkySpectroscopyIMP/ring-exp-toolkit/wiki).

## License
This repository is licensed under the **GNU GPLv3**.
