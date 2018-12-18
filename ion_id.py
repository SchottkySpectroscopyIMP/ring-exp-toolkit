#!/usr/bin/env python3
# −*− coding:utf-8 −*−

import io, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utility import Utility

class IID(Utility):
    '''
    A class for ion identification from a Schottky spectrum
    It can be imported into an IPython session to gain interactive operations
    '''

    def __init__(self, lpp_str, cen_freq, span, n_peak=10, L_CSRe=128.8, verbose=False):
        '''
        extract all the secondary fragments and their respective yields calculated by LISE++
        lpp_str:    LISE++ output file to be loaded
        cen_freq:   center frequency of the spectrum in MHz
        span:       span of the spectrum in kHz
        n_peak:     number of peaks to be identified
        L_CSRe:     circumference of CSRe in m, default value 128.8
        '''
        self.n_peak = n_peak
        super().__init__(cen_freq, span, L_CSRe, verbose)
        try:
            self.nucl_life = pd.read_csv("nuclear_half_lives.csv", na_filter=False)
            if self.verbose: print("nuclear half-lives loaded")
        except OSError:
            with io.StringIO() as buf:
                with open("nubase2016.txt") as nubase:
                    for line in nubase:
                        if line[7] != '0':
                            continue
                        element = ''.join(c for c in line[11:16] if c.isalpha())
                        stubs = line[60:78].split()
                        half_life = stubs[0].rstrip('#') if len(stubs) > 0 else "n/a"
                        if half_life[-1].isdigit():
                            half_life += ' '+stubs[1]
                        buf.write(','.join([line[:3], element, line[4:7], half_life]) + '\n')
                buf.seek(0) # rewind the file pointer to the beginning
                self.nucl_life = pd.read_csv(buf, na_filter=False, names=['A', "Element", 'Z', "HalfLife"])
            self.nucl_life.loc[self.nucl_life["Element"]=="Ed", "Element"] = "Nh" # Z=113
            self.nucl_life.loc[self.nucl_life["Element"]=="Ef", "Element"] = "Mc" # Z=115
            self.nucl_life.loc[self.nucl_life["Element"]=="Eh", "Element"] = "Ts" # Z=117
            self.nucl_life.loc[self.nucl_life["Element"]=="Ei", "Element"] = "Og" # Z=118
            self.nucl_life.to_csv("nuclear_half_lives.csv", index=False)
            if self.verbose: print("nuclear half-lives saved")
        with io.StringIO() as buf:
            with open(lpp_str, encoding="latin-1") as lpp:
                while True:
                    line = lpp.readline().strip()
                    if line == "[D1_DipoleSettings]":
                        self.Brho = float(lpp.readline().strip().split()[2]) # Tm
                    elif line == "[Calculations]":
                        break
                for line in lpp:
                    segment = line.strip().split(',')[0]
                    stubs = segment.split()
                    buf.write(' '.join([stubs[0]+stubs[1][:-1], stubs[-1][1:]]) + '\n')
            buf.seek(0)
            self.fragment = pd.read_csv(buf, delim_whitespace=True, names=["Ion", "Yield"])
        self.calc_peak()

    def calibrate_peak_loc(self, ion, peak_loc, harmonic):
        '''
        using the measured peak location with the identified ion to calibrate the magnetic rigidity of CSRe
        ion:        a string in the format of AElementQ, e.g., 3He2 
        peak_loc:   peak location in kHz after deduction of the center frequency
        harmonic:   harmonic number
        '''
        self.set_ion(ion)
        self.set_peak_loc(peak_loc, harmonic)
        self.calc_peak()

    def calibrate_rev_freq(self, ion, peak_loc_1, peak_loc_2):
        '''
        using the measured revolution frequency with the identified ion to calibrate the magnetic rigidity of CSRe
        ion:        a string in the format of AElementQ, e.g., 3He2 
        peak_loc_1: peak location in kHz after deduction of the center frequency
        peak_loc_2: another peak location belonging to the same ions but differed by one harmonic number
        '''
        rev_freq = np.absolute(peak_loc_2-peak_loc_1) / 1e3 # MHz
        self.set_ion(ion)
        self.set_rev_freq(rev_freq)
        self.calc_peak()

    def update_cen_freq(self, cen_freq):
        '''
        set a new center frequency of the spectrum in MHz
        '''
        self.cen_freq = cen_freq # MHz
        self.calc_peak()

    def update_span(self, span):
        '''
        set a new span of the spectrum in kHz
        '''
        self.span = span # kHz
        self.calc_peak()

    def update_n_peak(self, n_peak):
        '''
        set a new number of peaks to be shown in the output
        '''
        self.n_peak = n_peak
        self.show()

    def update_L_CSRe(self, L_CSRe):
        '''
        set the central orbital length of beams in CSRe in m
        '''
        self.L_CSRe = L_CSRe # m
        self.calc_peak()

    def calc_peak(self):
        '''
        calculate peak locations of the Schottky signals from secondary fragments visible in the pre-defined frequency range
        '''
        index, rev_freq, peak_loc, harmonic, half_life = [], [], [], [], []
        for row in self.fragment.itertuples():
            A, element, _ = re.split("([A-Z][a-z]?)", row.Ion)
            T_half = self.nucl_life.loc[(self.nucl_life['A']==int(A)) & (self.nucl_life["Element"]==element), "HalfLife"].item()
            self.set_ion(row.Ion)
            self.set_Brho(self.Brho)
            i = 0
            while i < self.peak_loc.size:
                index.append(row.Index)
                half_life.append(T_half)
                rev_freq.append(self.rev_freq) # MHz
                peak_loc.append(self.peak_loc[i]) # kHz
                harmonic.append(self.harmonic[i])
                i += 1
        candidate = self.fragment.iloc[index]
        candidate.index = np.arange(candidate.index.size)
        frequency = pd.DataFrame.from_dict({"HalfLife": half_life, "RevFreq": rev_freq, "PeakLoc": peak_loc, "Harmonic": harmonic})
        self.peak = pd.concat([candidate, frequency], axis=1)
        self.show()

    def show(self):
        '''
        list the most prominent peaks in a Schottky spectrum sorted in a descending order
        '''
        Q = self.peak["Ion"].str.split("[A-Z][a-z]?").str[-1].apply(int)
        self.peak["Weight"] = self.peak["Yield"] * Q**2 * self.peak["RevFreq"]**2
        self.peak.sort_values("Weight", ascending=False, inplace=True, kind="mergesort")
        print('-' * 16)
        print("center frequency\t{:g} MHz".format(self.cen_freq))
        print("span\t\t\t{:g} kHz".format(self.span))
        print("orbital length\t\t{:g} m".format(self.L_CSRe))
        print("Bρ\t\t\t{:.6g} Tm\n".format(self.Brho))
        print(self.peak.head(self.n_peak).to_string(index=False, justify="left",
            columns=["Weight", "Ion", "HalfLife", "Yield", "RevFreq", "PeakLoc", "Harmonic"],
            header=["Weight", " Ion", " Half-life", "Yield", "Rev.Freq.", "PeakLoc.", "Harmonic"],
            formatters=[
                lambda x: "{:<8.2e}".format(x),
                lambda x: "{:<7s}".format(x),
                lambda x: "{:<11s}".format(x),
                lambda x: "{:<9.2e}".format(x),
                lambda x: "{:<8.6f}".format(x),
                lambda x: "{:< 4.0f}".format(x),
                lambda x: "{:<3d}".format(x),
                ] ))

    def plot(self, display_num=10):
        '''
        plot the most prominent peaks in a Schottky spectrum
        '''
        self.update_n_peak(display_num) if display_num != -1 else self.update_n_peak(len(self.peak["Weight"]))
        print("total number of ions: {:}".format(len(self.peak["Weight"])))
        print("number of displayed ions: {:}".format(display_num)) if display_num != -1 else print("number of displayed ions: {:}".format(len(self.peak["Weight"])))
        peak_sort = self.peak.reset_index(drop=True)
        (markerline, stemlines, baseline) = plt.stem(peak_sort["PeakLoc"][:display_num], np.log10(peak_sort["Weight"][:display_num]), markerfmt='g.')
        plt.setp(baseline, color='grey', linewidth=1)
        plt.setp(stemlines, color='olive', linewidth=0.5)
        for i, ion in enumerate(peak_sort["Ion"][:display_num]):
            plt.text(peak_sort["PeakLoc"][i], np.log10(peak_sort["Weight"][i])+0.3, ion, fontsize=9)
        plt.xlabel("center frequency {:} [MHz]\nreference frequency [kHz]".format(self.cen_freq))
        plt.ylabel("Weight (log10)")
        plt.show()

    def find_ion(self, ion):
        '''
        show the information of the selected ion
        '''
        find_result = self.peak.set_index("Ion")
        find_result = find_result[~find_result.index.duplicated()].filter(like=ion, axis=0)
        find_result = find_result.reset_index()
        if find_result.empty:
            print("No valid ion!")
        else:
            print('-' * 16)
            print("center frequency\t{:g} MHz".format(self.cen_freq))
            print("span\t\t\t{:g} kHz".format(self.span))
            print("orbital length\t\t{:g} m".format(self.L_CSRe))
            print("Bρ\t\t\t{:.6g} Tm\n".format(self.Brho))
            print(find_result.head(self.n_peak).to_string(index=False, justify="left",
                columns=["Weight", "Ion", "HalfLife", "Yield", "RevFreq", "PeakLoc", "Harmonic"],
                header=["Weight", " Ion", " Half-life", "Yield", "Rev.Freq.", "PeakLoc.", "Harmonic"],
                formatters=[
                    lambda x: "{:<8.2e}".format(x),
                    lambda x: "{:<7s}".format(x),
                    lambda x: "{:<11s}".format(x),
                    lambda x: "{:<9.2e}".format(x),
                    lambda x: "{:<8.6f}".format(x),
                    lambda x: "{:< 4.0f}".format(x),
                    lambda x: "{:<3d}".format(x),
                    ] ))
        


if __name__ == "__main__":
    iid = IID("./58Ni28.lpp", 242.9, 500)
    iid.calibrate_peak_loc("58Ni28", -140, 161)
