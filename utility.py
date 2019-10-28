#!/usr/bin/env python3
# −*− coding:utf-8 −*−

import io
import numpy as np
import pandas as pd

class Utility(object):
    '''
    A class for the ion estimation during a beam time
    A useful tool to calculate the a special ion (bare/H-like/He-like)
    It is intended to be imported, e.g., into an IPython session
    A preparation for the ion identification of the Schottky spectrum at CSRe
    '''
    c = 299792458 # speed of light in m/s
    e = 1.6021766208e-19 # elementary charge in C
    me = 5.4857990907e-4 # electron mass in u
    u2kg = 1.66053904e-27 # amount of kg per 1 u
    MeV2u = 1.073544216e-3 # amount of u per 1 MeV

    def __init__(self, cen_freq, span, L_CSRe=128.8, verbose=True):
        '''
        load the atomic mass dataset from disk, if any, otherwise build it and save it to disk
        (atomic data including Mass, Half-life of all the nuclides in the Bare, H-like and, He-like ionization)
        cen_freq:   center frequency of the spectrum in MHz
        span:       span of the spectrum in kHz
        L_CSRe:     circumference of CSRe in m, default value 128.8

        mass data from AME2016
        half life from NUBASE2016
        eletron binding energy from NIST Atomic Spectra Database - Ionization Energies Form 
        https://physics.nist.gov/PhysRefData/ASD/ionEnergy.html
        '''
        self.cen_freq = cen_freq # MHz
        self.span = span # kHz
        self.L_CSRe = L_CSRe # m
        self.verbose = verbose
        try:
            self.atom_data = pd.read_csv("atomic_data.csv")
            if self.verbose: print("atomic data loaded")
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
                atom_mass = pd.read_csv(buf, delim_whitespace=True, names=['A', "Element", 'Z', "Mass", "Source"])
            with io.StringIO() as buf:
                with open("./nubase2016.txt") as nubase:
                    for line in nubase:
                        if line[7] != '0':
                            continue
                        stubs = line[60:78].split()
                        half_life = stubs[0].rstrip('#') if len(stubs) > 0 else "n/a"
                        half_life += '' + stubs[1] if half_life[-1].isdigit() else ""
                        buf.write(','.join([line[:3], line[4:7], half_life]) + '\n')
                buf.seek(0)
                nucl_life = pd.read_csv(buf, na_filter=False, names=['A', 'Z', "HalfLife"])
            atom_data = pd.merge(atom_mass, nucl_life)
            atom_data.loc[atom_data["Element"]=="Ed", "Element"] = "Nh" # Z=113
            atom_data.loc[atom_data["Element"]=="Ef", "Element"] = "Mc" # Z=115
            atom_data.loc[atom_data["Element"]=="Eh", "Element"] = "Ts" # Z=117
            atom_data.loc[atom_data["Element"]=="Ei", "Element"] = "Og" # Z=118
            atom_data["Mass"] /= 1e6 # masses are scaled to be in u
            atom_data['Q'] = atom_data['Z']
            atom_data["Mass"] -= self.me * atom_data['Q'] # ionic mass in u for bare ions
            del atom_mass, nucl_life
            self.atom_data = atom_data
            bde = pd.read_csv("./electron_binding_energy.csv", encoding='utf8')
            bde["BindingEnergy"] *= self.MeV2u *1e-3        
            def ion_like(pre_ion, bindEn, electron_num):
                if pre_ion["Element"] in bindEn["Element"].values:
                    return (pre_ion["Mass"] + electron_num * self.me - bindEn[bindEn["Element"]==pre_ion["Element"]]["BindingEnergy"].values)[0]
                else:
                    return np.nan
            for electron_num in [1, 2]:
                nucl_like = atom_data[atom_data['Z'] > electron_num].reset_index(drop=True)
                nucl_like["Mass"] = nucl_like.apply(lambda x: ion_like(x, bde, electron_num), axis=1)
                nucl_like['Q'] -= electron_num
                nucl_like["HalfLife"] += " {:s}".format("*"*electron_num) # * notes for H-like, ** notes for He-like
                self.atom_data = pd.concat([self.atom_data, nucl_like], ignore_index=True)
            self.atom_data.to_csv("atomic_data.csv", index=False)
            if self.verbose: print("atomic data saved")

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
        select_A_El = (self.atom_data['A']==A) & (self.atom_data["Element"]==element)
        if not select_A_El.any():
            print("Error: ion is too rare!")
            return
        if Q not in self.atom_data[select_A_El]['Q'].values:
            print("Error: ion is not bare or H-like or He-like. Change to the bare ion instead.")
            Q = self.atom_data[select_A_El]['Q'].max()
        self.index = self.atom_data.loc[((self.atom_data['A']==A) & (self.atom_data["Element"]==element) & (self.atom_data['Q']==Q))].index[0] # index of the target ion in the lookup table
        self.Q = Q
        self.mass = self.atom_data.iloc[self.index]["Mass"]
        self.halfLife = self.atom_data.iloc[self.index]["HalfLife"]

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
        print("target ion\t\t{0.A:d}{0.Element:s}{1:d}+".format(self.atom_data.iloc[self.index], self.Q))
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
        print("half life\t\t{:s}".format(self.halfLife))
    
    def help(self):
        '''
        display all the available functions of the class: Utility
        '''
        print('--' * 10 + '\n')
        print('Display all avaliable functions of the Utility\n')
        print("Input Only:")
        print("set_ion(ion)\n\tset the target ion, to be input in the format of AEmlementQ, e.g., 3H2")
        print("set_cen_freq(cen_freq)\n\tset a new center frequency of the spectrum [MHz]")
        print("set_span(span)\n\tset a new span of the spectrum [kHz]")
        print("set_L_CSRe(L_CSRe)\n\tset the adjusted circumference of CSRe [m]")
        print("Display the estimation result after input:")
        print("set_energy(energy)\n\tset the kinetic energy the target ion [MeV/u]")
        print("set_gamma(gamma)\n\tset the Lorentz factor of the target ion")
        print("set_beta(beta)\n\tset the velocity of the target ion in unit of speed of light")
        print("set_Brho(Brho)\n\tset the magnetic rigidity of the target ion [Tm]")
        print("set_rev_freq(rev_freq)\n\tset the revolution frequency of the target ion [MHz]")
        print("set_peak_loc(peak_loc, harmonic)\n\tset the peak location [kHz] of the target ion and corresponding harmonic for the calibration")
        print('\n' + '--' * 10) 


if __name__ == "__main__":
    utility = Utility(242.9, 500)
    utility.set_ion("58Ni19")
    utility.set_energy(143.92)
