import os
from ase.data import atomic_numbers
# Table (1) of
# D. WAASMAIER AND A. KIRFEL, Acta Cryst. (1995). A51, 416-431
# 10.1107/s0108767394013292


def read_waasmaier_coeffs(symbols_list):
    coeffs_dict = {}
    symbols_and_atomic_numbers: list = [
        (symbol, atomic_numbers[symbol]) for symbol in symbols_list
    ]

    # Sort by atomic numbers for later
    symbols_and_atomic_numbers = sorted(symbols_and_atomic_numbers, key=lambda x: x[1])
    datafilepath = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "waasmaier_coefficients.dat"
    )
    with open(datafilepath) as fd:
        for symbol, atomic_number in symbols_and_atomic_numbers:
            for line in fd:
                if line.strip() == f"#S  {atomic_number}  {symbol}":
                    next(fd)
                    next(fd)
                    coeffs_dict[symbol] = [
                        float(x) for x in fd.readline().strip().split()
                    ]
                    break
    return coeffs_dict
