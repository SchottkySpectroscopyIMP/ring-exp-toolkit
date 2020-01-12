#!/usr/bin/env python3
# −*− coding:utf-8 −*−

import io
import numpy as np
import pandas as pd

class Utility(object):
    '''
    A class encapsulating some utility methods for the quick calculation during an isochronous Schottky beam time.
    It is a useful tool to calculate a specific ion (bare/H-like/He-like/Li-like).
    It is intended to be imported, e.g., into an IPython session, to be used for the ion identification.
    '''

    c = 299792458 # speed of light in m/s
    e = 1.6021766208e-19 # elementary charge in C
    me = 5.4857990907e-4 # electron mass in u
    u2kg = 1.66053904e-27 # amount of kg per 1 u
    MeV2u = 1.073544216e-3 # amount of u per 1 MeV
    orb_e = {0: "bare", 1: "H-like", 2: "He-like", 3: "Li-like"} # ionic charge state related to its orbital electron count


    def __init__(self, cen_freq, span, L_CSRe=128.8, verbose=True):
        '''
        Load the ionic data from disk, if any, otherwise build it and save it to disk.
        The ionic data contain masses of nuclides in the bare, H-like, He-like, Li-like, and neutral charge states.
        Besides, the half-lives of the corresponding atoms are included as a reference.

        cen_freq:   center frequency of the spectrum in MHz
        span:       span of the spectrum in kHz
        L_CSRe:     circumference of CSRe in m, default value 128.8
        '''
        self.cen_freq = cen_freq # MHz
        self.span = span # kHz
        self.L_CSRe = L_CSRe # m
        self.verbose = verbose
        try:
            self.ion_data = pd.read_csv("ionic_data.csv")
            if self.verbose: print("ionic data loaded")
        except OSError:
            with io.StringIO() as buf:
                with open("./mass16.txt") as ame:
                    for _ in '_'*39:
                        ame.readline()
                    for l in ame:
                        if l[106] == '#':
                            buf.write(l[16:23] + "0 " + l[11:15] + l[96:99] + '.' + l[100:106] + " estimated\n")
                        else:
                            buf.write(l[16:23] + "0 " + l[11:15] + l[96:99] + '.' + l[100:106] + l[107:112] + " measured\n")
                buf.seek(0) # rewind the file pointer to the beginning
                atom_mass = pd.read_csv(buf, delim_whitespace=True, names=['A', "Element", 'Q', 'Z', "Mass", "Source"])
                atom_mass.loc[atom_mass["Element"]=="Ed", "Element"] = "Nh" # Z=113
                atom_mass.loc[atom_mass["Element"]=="Ef", "Element"] = "Mc" # Z=115
                atom_mass.loc[atom_mass["Element"]=="Eh", "Element"] = "Ts" # Z=117
                atom_mass.loc[atom_mass["Element"]=="Ei", "Element"] = "Og" # Z=118
            with io.StringIO() as buf:
                with open("./nubase2016.txt") as nubase:
                    for line in nubase:
                        if line[7] != '0':
                            continue
                        stubs = line[60:78].split()
                        half_life = stubs[0].rstrip('#') if len(stubs) > 0 else "n/a"
                        half_life += ' ' + stubs[1] if half_life[-1].isdigit() else ''
                        buf.write(','.join([line[:3], line[4:7], half_life]) + '\n')
                buf.seek(0)
                atom_half_life = pd.read_csv(buf, na_filter=False, names=['A', 'Z', "Half-Life"])
            neutral = pd.merge(atom_mass, atom_half_life)
            del atom_mass, atom_half_life
            ioniz_eng = pd.read_csv("./ionization.csv", comment='#')
            bare = neutral[neutral['Z']!=0].copy() # ignore neutron
            bare['Q'] = bare['Z']
            bare["Mass"] -= self.me*bare['Q'] - (14.4381*bare['Q']**2.39+1.55468e-6*bare['Q']**5.35)/1e6*self.MeV2u
            h_like = bare[bare['Q']!=1].copy() # ignore isotopes of hydrogen
            h_like['Q'] -= 1
            h_like["Half-Life"] += " *"
            h_like = h_like.merge(ioniz_eng, on=["Element", 'Q'])
            h_like["Mass"] += self.me - h_like["Ionization"]/1e6*self.MeV2u
            h_like.drop(columns="Ionization", inplace=True)
            he_like = h_like[h_like['Q']!=1].copy() # ignore isotopes of helium
            he_like['Q'] -= 1
            he_like["Half-Life"] += '*'
            he_like = he_like.merge(ioniz_eng, on=["Element", 'Q'])
            he_like["Mass"] += self.me - he_like["Ionization"]/1e6*self.MeV2u
            he_like.drop(columns="Ionization", inplace=True)
            li_like = he_like[he_like['Q']!=1].copy() # ignore isotopes of lithium
            li_like['Q'] -= 1
            li_like["Half-Life"] += '*'
            li_like = li_like.merge(ioniz_eng, on=["Element", 'Q'])
            li_like["Mass"] += self.me - li_like["Ionization"]/1e6*self.MeV2u
            li_like.drop(columns="Ionization", inplace=True)
            self.ion_data = pd.concat([bare, h_like, he_like, li_like, neutral])
            self.ion_data.sort_values(by=['A', 'Z', 'Q'], axis=0, inplace=True)
            self.ion_data.reset_index(drop=True, inplace=True)
            self.ion_data.to_csv("ionic_data.csv", index=False)
            if self.verbose: print("ionic data saved")

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
        select_A_El = (self.ion_data['A']==A) & (self.ion_data["Element"]==element)
        if not select_A_El.any():
            print("Error: ion is too rare!")
            raise SystemExit
        if Q not in self.ion_data[select_A_El]['Q'].values:
            print("Warning: ion is not found with the given charge state, change to the bare ion instead.")
            Q = self.ion_data[select_A_El]['Q'].max()
        self.index = self.ion_data[select_A_El & (self.ion_data['Q']==Q)].index[0] # index of the target ion in the lookup table
        self.Q, self.mass, self.half_life = Q, self.ion_data.loc[self.index]["Mass"], self.ion_data.loc[self.index]["Half-Life"]

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
        print("target ion\t\t{0.A:d}{0.Element:s}{1:d}+".format(self.ion_data.iloc[self.index], self.Q))
        print("charge state\t\t{:s}".format(self.orb_e[self.ion_data.loc[self.index]['Z']-self.Q]))
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
        print("atomic half-life\t{:s}".format(self.half_life))
    
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
