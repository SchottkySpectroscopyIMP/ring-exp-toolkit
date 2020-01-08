#!/usr/bin/env python3
# −*− coding:utf-8 −*−

import io, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utility import Utility

class IID(Utility):
    '''
    A script for auto calibrating the ion identification result based on the input Schottky spectrum
    '''
    def __init__(self, lppion, cen_freq, span, n_peak=10, GUI_mode=False, L_CSRe=128.8, verbose=False):
        '''
        extract all the secondary fragments and their respective yields calculated by LISE++
        (including Mass, Half-life, Yield of all the fragments)
        lppion:    LISE++ output file to be loaded
        cen_freq:   center frequency of the spectrum in MHz
        span:       span of the spectrum in kHz
        n_peak:     number of peaks to be identified
        L_CSRe:     circumference of CSRe in m, default value 128.8
        '''
        self.n_peak = n_peak
        self.sigma = 1.0e-6
        self.GUI_mode = GUI_mode
        super().__init__(cen_freq, span, L_CSRe, verbose)
        with io.StringIO() as buf:
            with open(lppion, encoding='latin-1') as lpp:
                while True:
                    line = lpp.readline().strip()
                    if line == "[D1_DipoleSettings]":
                        self.Brho = float(lpp.readline().strip().split()[2]) # Tm
                    elif line == "[Calculations]":
                        break
                for line in lpp:
                    segment = line.strip().split(',')[0]
                    stubs = segment.split()
                    A, element, Q = re.split("([A-Z][a-z]?)", stubs[0]+stubs[-2][:-1])
                    buf.write(' '.join([A, element, Q, stubs[-1][1:]]) + '\n')
            buf.seek(0)
            self.fragment = pd.read_csv(buf, delim_whitespace=True, names=['A', "Element", 'Q', "Yield"])
        self.fragment = pd.merge(self.fragment, self.atom_data)
        self.calc_peak()

    def calc_peak(self):
        '''
        calculate peak locations of the Schottky signals from secondary fragments visible in the pre-defined frequency range
        '''
        self.peak = self.fragment
        gamma_beta = self.Brho / self.peak["Mass"] * self.peak['Q'] / self.c / self.u2kg * self.e
        beta = gamma_beta / np.sqrt(1 + gamma_beta**2)
        gamma = 1 / np.sqrt(1 - beta**2)
        energy = (gamma - 1) / self.MeV2u # MeV/u
        self.peak["RevFreq"] = beta * self.c / self.L_CSRe / 1e6 # MHz
        self.peak["Weight"] = self.peak["Yield"] * self.peak['Q']**2 * self.peak["RevFreq"]**2
        lower_freq, upper_freq = self.cen_freq - self.span/2e3, self.cen_freq + self.span/2e3 # MHz, MHz
        def calc_harmonic(x, lower_freq, upper_freq):
            return np.arange(np.ceil(lower_freq/x), np.floor(upper_freq/x)+1).astype(int)
        self.peak["Harmonic"] = self.peak.apply(lambda x: calc_harmonic(x["RevFreq"], lower_freq, upper_freq), axis=1)
        peak_dict_temp = self.peak.to_dict()
        peak_dict = {'A': {}, "Element": {}, 'Q': {}, "Yield": {}, "HalfLife": {}, "RevFreq": {}, "Harmonic": {}, "PeakLoc": {}, "Weight": {}}
        ind = 0
        for index in range(len(peak_dict_temp["Harmonic"])):
            for h in peak_dict_temp["Harmonic"][index]:
                for each in peak_dict:
                    if each is "Harmonic":
                        peak_dict[each].update({ind: h})
                    elif each is "PeakLoc":
                        peak_dict[each].update({ind: (h * peak_dict_temp["RevFreq"][index] - self.cen_freq) * 1e3})
                    else:
                        peak_dict[each].update({ind: peak_dict_temp[each][index]})
                ind += 1
        self.peak = pd.DataFrame.from_dict(peak_dict)
        self.peak["Ion"] = self.peak.apply(lambda x: "{}{}{}".format(x['A'], x["Element"], x['Q']), axis=1)
        self.peak.sort_values("Weight", ascending=False, inplace=True, kind="mergesort")
        self.peak.reset_index(drop=True, inplace=True)
        if self.GUI_mode:
            frequencyRange, ionPeaks = self.calc_gaussian_peak()
            return frequencyRange, ionPeaks, self.peak.loc[:, ["Ion", "Yield", "HalfLife", "Harmonic", "PeakLoc", "RevFreq", "Weight"]]
        else:
            self.show()
            return

    def calc_gaussian_peak(self):
        '''
        return the spectrum including all the selected ions in form of Gaussian peak
        default: sigma = delta f / rev_freq = 1e-5
        for each ion: width = sigma * harmonic * rev_freq / 1.66
        '''
        frequencyRange = np.linspace(-self.span/2, self.span/2, 8192) # kHz
        lim = self.peak["Weight"].max() / 1e5
        def calc_ion_peak(x):
            width = self.sigma * x["Harmonic"] * x["RevFreq"] * 1e3 / 1.66
            a = x["Weight"] / (np.sqrt(2 * np.pi) * width) * np.exp(-(frequencyRange - x["PeakLoc"])**2 / (2 * width**2))
            a[a < lim] = lim
            return a
        ionPeaks = np.sum(self.peak.apply(lambda x: calc_ion_peak(x), axis=1))
        return frequencyRange, ionPeaks
        
    def set_ion(self, ion):
        '''
        override the function from the Utility.set_ion()
        set the target ion, to be input in the format of AElementQ, e.g., 3He2
        '''
        element = ''.join(c for c in ion if c.isalpha())
        A, Q = map(int, ion.split(element))
        select_nucl = (self.fragment['A']==A) & (self.fragment["Element"]==element) & (self.fragment["Q"]==Q)
        if not select_nucl.any():
            print("Error: ion is not existed in the fragments!")
            return
        self.index = self.fragment.loc[select_nucl].index[0] # index of the target ion in the lookup table
        self.Q = Q
        self.mass = self.fragment.iloc[self.index]["Mass"]
        self.halfLife = self.fragment.iloc[self.index]["HalfLife"]

    def calibrate_Brho(self, Brho):
        '''
        using the measured Brho with the identified ion to calibrate
        Brho:       the magnetic rigidity of the target ion in Tm
        '''
        self.Brho = Brho
        return self.calc_peak()

    def calibrate_peak_loc(self, ion, peak_loc, harmonic):
        '''
        using the measured peak location with the identified ion to calibrate the magnetic rigidity of CSRe
        ion:        a string in the format of AElementQ, e.g., 3He2 
        peak_loc:   peak location in kHz after deduction of the center frequency
        harmonic:   harmonic number
        '''
        rev_freq = (self.cen_freq + peak_loc/1e3) / harmonic # MHz
        self.set_ion(ion)
        self.set_rev_freq(rev_freq)
        return self.calc_peak()
    
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
        return self.calc_peak()

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

    def show(self):
        '''
        list the most prominent peaks in a Schottky spectrum sorted in a descending order
        '''
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

    def help(self):
        '''
        override the function from Utility.help() 
        display all the available functions of the class: IID
        '''
        print('--' * 10 + '\n')
        print('Display all avaliable functions of the IID\n')
        print("calibrate_Brho(Brho)\n\tusing the measured Brho with the identified ion to calibrate\n\tBrho: the magnetic rigidity of the target ion [Tm]")
        print("calibrate_peak_loc(ion, peak_loc, harmonic)\n\tusing the measured peak location with the identified ion to calibrate the magnetic rigidity of CSRe\n\tion:\t\ta string in the format of AElementQ, e.g., 3H2\n\tpeak_loc:\tpeak location after deduction of the center frequency [kHz]\n\tharmonic:\tharmonic number")
        print("calibrate_rev_freq(ion, peak_loc_1, peak_loc_2)\n\tion:\t\ta string in the format of AElementQ, e.g., 3H2\n\tpeak_loc_1:\tpeak location after deduction of the center frequency [kHz]\n\tpeak_loc_2:\tanother peak location belonging to the same ions but differed by one harmonic number [kHz]")
        print("update_cen_freq(cen_freq)\n\tset a new center frequency of the spectrum [MHz]")
        print("update_span(span)\n\tset a new span of the spectrum [kHz]")
        print("update_n_peak(n_peak)\n\tset a new numer of peaks to be shown in the output")
        print("update_L_CSRe(L_CSRe)\n\tset the adjusted circumference of CSRe [m]")
        print('\n' + '--' * 10) 

if __name__ == "__main__":
    iid = IID("./58Ni28.lpp", 242.9, 500)
    iid.calibrate_peak_loc("58Ni28", -140, 161)
