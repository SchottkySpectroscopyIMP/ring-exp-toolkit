#!/usr/bin/env python3
# −*− coding:utf-8 −*−

import io
import numpy as np
import pandas as pd

class Utility(object):
    '''
    A class encapsulating some utility methods for the quick calculation during a beam time
    It is intended to be imported, for example, into an IPython session
    '''

    c = 299792458 # speed of light in m/s
    e = 1.6021766208e-19 # elementary charge in C
    me = 5.4857990907e-4 # electron mass in u
    u2kg = 1.66053904e-27 # amount of kg per 1 u
    MeV2u = 1.073544216e-3 # amount of u per 1 MeV

    def __init__(self, cen_freq, span, L_CSRe=128.8, verbose=True):
        '''
        load the atomic mass dataset from disk, if any, otherwise build it and save it to disk
        cen_freq:   center frequency of the spectrum in MHz
        span:       span of the spectrum in kHz
        L_CSRe:     circumference of CSRe in m, default value 128.8
        '''
        self.cen_freq = cen_freq # MHz
        self.span = span # kHz
        self.L_CSRe = L_CSRe # m
        self.verbose = verbose
        try:
            self.atom_mass = pd.read_csv("atomic_masses.csv")
            if self.verbose: print("atomic masses loaded")
        except OSError:
            with io.StringIO() as buf:
                with open("./mass16.txt") as ame:
                    for _ in '_'*39:
                        ame.readline()
                    for l in ame:
                        if l[106] == '#':
                            buf.write(l[16:23] + l[11:15] + l[96:99] + l[100:106] + " estimated\n")
                        else:
                            buf.write(l[16:23] + l[11:15] + l[96:99] + l[100:112] + " measured\n")
                buf.seek(0) # rewind the file pointer to the beginning
                self.atom_mass = pd.read_csv(buf, delim_whitespace=True, names=['A', "Element", 'Z', "Mass", "Source"])
            self.atom_mass.loc[self.atom_mass["Element"]=="Ed", "Element"] = "Nh" # Z=113
            self.atom_mass.loc[self.atom_mass["Element"]=="Ef", "Element"] = "Mc" # Z=115
            self.atom_mass.loc[self.atom_mass["Element"]=="Eh", "Element"] = "Ts" # Z=117
            self.atom_mass.loc[self.atom_mass["Element"]=="Ei", "Element"] = "Og" # Z=118
            self.atom_mass["Mass"] /= 1e6 # masses are scaled to be in u
            self.atom_mass.to_csv("atomic_masses.csv", index=False)
            if self.verbose: print("atomic masses saved")

    def set_cen_freq(self, cen_freq):
        '''
        set a new center frequency of the spectrum in MHz
        '''
        self.cen_freq = cen_freq # MHz

    def set_span(self, span):
        '''
        set a new span of the spectrum in kHz
        '''
        self.span = span # kHz

    def set_L_CSRe(self, L_CSRe):
        '''
        set the adjusted circumference of CSRe in m
        '''
        self.L_CSRe = L_CSRe # m

    def set_ion(self, ion):
        '''
        set the target ion, to be input in the format of AElementQ, e.g., 3He2
        '''
        element = ''.join(c for c in ion if c.isalpha())
        A, Q = map(int, ion.split(element))
        select_A_El = (self.atom_mass['A']==A) & (self.atom_mass["Element"]==element)
        if not select_A_El.any():
            print("Error: ion is too rare!")
            return
        self.index = self.atom_mass.loc[select_A_El].index[0] # index of the target ion in the lookup table
        self.Q = Q if self.atom_mass.iloc[self.index]['Z'] >= Q else self.atom_mass.iloc[self.index]['Z'] # at most fully stripped
        self.mass = self.atom_mass.iloc[self.index]["Mass"] - self.me * self.Q # ionic mass in u after deducting the masses of the stripped electrons

    def set_gamma(self, gamma):
        '''
        set the Lorentz factor of the target ion
        '''
        self.gamma = gamma
        self.beta = np.sqrt(1 - self.gamma**-2)
        self.Brho = self.gamma * self.beta * self.mass / self.Q * self.c * self.u2kg / self.e # magnetic rigidity in Tm
        self.energy = (self.gamma - 1) / self.MeV2u # kinetic energy in MeV/u
        self.rev_freq = self.beta * self.c / self.L_CSRe / 1e6 # revolution frequency in MHz
        lower_freq, upper_freq = self.cen_freq - self.span/2e3, self.cen_freq + self.span/2e3 # MHz, MHz
        self.harmonic = np.arange(np.ceil(lower_freq/self.rev_freq), np.floor(upper_freq/self.rev_freq)+1).astype(int) # selected harmonic numbers of the peaks viewed through the frequency window
        self.peak_loc = (self.harmonic * self.rev_freq - self.cen_freq) * 1e3  # peak locations in kHz after deduction of the center frequency
        if self.verbose: self.show()

    def set_beta(self, beta):
        '''
        set the velocity of the target ion in unit of speed of light
        '''
        self.beta = beta
        self.gamma = 1 / np.sqrt(1 - self.beta**2)
        self.Brho = self.gamma * self.beta * self.mass / self.Q * self.c * self.u2kg / self.e # Tm
        self.energy = (self.gamma - 1) / self.MeV2u # MeV/u
        self.rev_freq = self.beta * self.c / self.L_CSRe / 1e6 # MHz
        lower_freq, upper_freq = self.cen_freq - self.span/2e3, self.cen_freq + self.span/2e3 # MHz, MHz
        self.harmonic = np.arange(np.ceil(lower_freq/self.rev_freq), np.floor(upper_freq/self.rev_freq)+1).astype(int)
        self.peak_loc = (self.harmonic * self.rev_freq - self.cen_freq) * 1e3  # kHz
        if self.verbose: self.show()

    def set_Brho(self, Brho):
        '''
        set the magnetic rigidity of the target ion in Tm
        '''
        self.Brho = Brho # Tm
        gamma_beta = self.Brho / self.mass * self.Q / self.c / self.u2kg * self.e
        self.beta = gamma_beta / np.sqrt(1 + gamma_beta**2)
        self.gamma = 1 / np.sqrt(1 - self.beta**2)
        self.energy = (self.gamma - 1) / self.MeV2u # MeV/u
        self.rev_freq = self.beta * self.c / self.L_CSRe / 1e6 # MHz
        lower_freq, upper_freq = self.cen_freq - self.span/2e3, self.cen_freq + self.span/2e3 # MHz, MHz
        self.harmonic = np.arange(np.ceil(lower_freq/self.rev_freq), np.floor(upper_freq/self.rev_freq)+1).astype(int)
        self.peak_loc = (self.harmonic * self.rev_freq - self.cen_freq) * 1e3  # kHz
        if self.verbose: self.show()

    def set_energy(self, energy):
        '''
        set the kinetic energy of the target ion in MeV/u
        '''
        self.energy = energy # MeV/u
        self.gamma = 1 + self.energy * self.MeV2u
        self.beta = np.sqrt(1 - self.gamma**-2)
        self.Brho = self.gamma * self.beta * self.mass / self.Q * self.c * self.u2kg / self.e # Tm
        self.rev_freq = self.beta * self.c / self.L_CSRe / 1e6 # MHz
        lower_freq, upper_freq = self.cen_freq - self.span/2e3, self.cen_freq + self.span/2e3 # MHz, MHz
        self.harmonic = np.arange(np.ceil(lower_freq/self.rev_freq), np.floor(upper_freq/self.rev_freq)+1).astype(int)
        self.peak_loc = (self.harmonic * self.rev_freq - self.cen_freq) * 1e3  # kHz
        if self.verbose: self.show()

    def set_rev_freq(self, rev_freq):
        '''
        set the revolution frequency of the target ion in MHz
        '''
        self.rev_freq = rev_freq # MHz
        self.beta = self.rev_freq * self.L_CSRe / self.c * 1e6
        self.gamma = 1 / np.sqrt(1 - self.beta**2)
        self.Brho = self.gamma * self.beta * self.mass / self.Q * self.c * self.u2kg / self.e # Tm
        self.energy = (self.gamma - 1) / self.MeV2u # MeV/u
        lower_freq, upper_freq = self.cen_freq - self.span/2e3, self.cen_freq + self.span/2e3 # MHz, MHz
        self.harmonic = np.arange(np.ceil(lower_freq/self.rev_freq), np.floor(upper_freq/self.rev_freq)+1).astype(int)
        self.peak_loc = (self.harmonic * self.rev_freq - self.cen_freq) * 1e3  # kHz
        if self.verbose: self.show()

    def set_peak_loc(self, peak_loc, harmonic):
        '''
        set the peak location of the target ion in kHz after deduction of the center frequency
        in order to unambiguously deduce the revolution frequency, the harmonic number must also be specified
        '''
        self.rev_freq = (self.cen_freq + peak_loc/1e3) / harmonic # MHz
        self.beta = self.rev_freq * self.L_CSRe / self.c * 1e6
        self.gamma = 1 / np.sqrt(1 - self.beta**2)
        self.Brho = self.gamma * self.beta * self.mass / self.Q * self.c * self.u2kg / self.e # Tm
        self.energy = (self.gamma - 1) / self.MeV2u # MeV/u
        lower_freq, upper_freq = self.cen_freq - self.span/2e3, self.cen_freq + self.span/2e3 # MHz, MHz
        self.harmonic = np.arange(np.ceil(lower_freq/self.rev_freq), np.floor(upper_freq/self.rev_freq)+1).astype(int)
        self.peak_loc = (self.harmonic * self.rev_freq - self.cen_freq) * 1e3  # kHz
        if self.verbose: self.show()

    def show(self):
        '''
        print all the kinematic and spectroscopic parameters of the target ion
        '''
        print('-' * 10)
        print("target ion\t\t{0.A:d}{0.Element:s}{1:d}+".format(self.atom_mass.iloc[self.index], self.Q))
        print("γ\t\t\t{:.6g}".format(self.gamma))
        print("β\t\t\t{:.6g}".format(self.beta))
        print("Bρ\t\t\t{:.6g} Tm".format(self.Brho))
        print("kinetic energy\t\t{:.6g} MeV/u".format(self.energy))
        print("ring circumference\t{:g} m".format(self.L_CSRe))
        print("revolution frequency\t{:.6g} MHz".format(self.rev_freq))
        print("center frequency\t{:g} MHz".format(self.cen_freq))
        print("span\t\t\t{:g} kHz".format(self.span))
        print("peak location(s)\t" + ', '.join(["{:.0f}".format(item) for item in self.peak_loc]) + " kHz")
        print("harmonic number(s)\t" + ', '.join(["{:d}".format(item) for item in self.harmonic]))


if __name__ == "__main__":
    utility = Utility(242.9, 500)
    utility.set_ion("58Ni39")
    utility.set_energy(143.92)
