#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys, re, io
import numpy as np
import pandas as pd
import pyqtgraph as pg

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from utility import Utility
from pandas_model import PandasModel
from multithread import Worker

class IID(Utility):
    '''
    A class for ion identification from a Schottky spectrum
    It can be imported into an GUI based on the PyQt5
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
        self.sigma = 5.0E-06 / 5.0
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
                rev_freq.append(self.rev_freq)    # MHz
                peak_loc.append(self.peak_loc[i]) # kHz
                harmonic.append(self.harmonic[i])
                i += 1
        candidate = self.fragment.iloc[index]
        candidate.index = np.arange(candidate.index.size)
        frequency = pd.DataFrame.from_dict({"HalfLife": half_life, "RevFreq": rev_freq, "PeakLoc": peak_loc, "Harmonic": harmonic})
        self.peak = pd.concat([candidate, frequency], axis=1)
        Q = self.peak["Ion"].str.split("[A-Z][a-z]?").str[-1].apply(int)
        self.peak["Weight"] = self.peak["Yield"] * Q**2 * self.peak["RevFreq"]**2
        self.peak.sort_values("Weight", ascending=False, inplace=True, kind="mergesort")
        self.peak = self.peak.reset_index(drop=True)
        #self.peak.style.format({"Ion": "{:<7s}", "Yield": "{:<9.2e}", "HalfLife": "{:<11s}", "Harmonic": {:<3d}", "PeakLoc": "{:<4.0f}", "RevFreq": "{:<8.6f}", "Weight": "{:<8.2e}"})
        #print(self.peak)
        return self.peak

    def calibrate_peak_loc(self, ion, peak_loc, harmonic):
        '''
        using the measured peak location with the identified ion to calibrate
        ion:        a string in the format of AElementQ, e.g. 3He2
        peak_loc:   peak location in kHz after deduction of the center frequency
        harmonic:   harmonic number
        '''
        self.set_ion(ion)
        self.set_peak_loc(peak_loc, harmonic)
        self.peak = self.calc_peak()
        return self.peak

    def calibrate_Brho(self, Brho):
        '''
        using the measured Brho with the identified ion to calibrate
        Brho:       the magnetic rigidity of the target ion in Tm
        '''
        self.set_Brho(Brho)
        self.peak = self.calc_peak()
        return self.peak

    def find_isotopes(self, element_input):
        '''
        show the all isotopes' information of the selected element
        '''
        if element_input == "": # show the highest Weight ion
            try:
                return re.sub("[^A-Za-z]", "", self.peak_sort["Ion"][0]), self.peak_sort
            except:
                return re.sub("[^A-Za-z]", "", self.peak["Ion"][0]), self.peak
        self.isotopes_result = self.peak_sort.copy()
        self.isotopes_result = self.isotopes_result[self.isotopes_result["Ion"].str.contains(r"\d+" + element_input + "+\d")]
        self.isotopes_result = self.isotopes_result.reset_index(drop=True)
        if self.isotopes_result.empty:
            return element_input, pd.DataFrame(columns=['Ion', 'Yield', 'HalfLife', 'Harmonic', 'PeakLoc', 'RevFreq', 'Weight'])
        else:
            return re.sub("[^A-Za-z]", "", self.isotopes_result["Ion"][0]), self.isotopes_result

    def find_ion(self, peakLoc_input):
        '''
        show the information of the selected ion peak
        '''
        ion_result = self.peak
        ion_result = ion_result[np.abs(ion_result.PeakLoc-peakLoc_input) <= (self.sigma*self.cen_freq*1E+03)]
        ion_result = ion_result.reset_index(drop=True)
        return ion_result

    def gaussian_peak(self, display_ion):
        '''
        return the data for the gaussian peaks of all the ions, and the data for the gaussian peaks of the selected element
        buttom_width = 5 * sigma   # using gaussian
        '''
        self.peak_sort = self.peak.copy()
        frequency_range = np.arange(-self.span/2, self.span/2, 0.01)
        lim = np.max(self.peak_sort["Weight"]) / 1.0E+5
        def formFunc(row_Weight, row_PeakLoc):
            a = row_Weight / (np.sqrt(2*np.pi) * self.sigma*(self.cen_freq * 1.0E+03 + row_PeakLoc)) * np.exp(-(frequency_range - row_PeakLoc)**2 / (2 * (self.sigma * (self.cen_freq * 1.0E+03 + row_PeakLoc))**2))
            a[a < lim] = lim
            return a
        self.peak_sort["PeakFunc"] = self.peak_sort.apply(lambda row: formFunc(row['Weight'], row['PeakLoc']), axis=1)
        peak_sum = self.peak_sort["PeakFunc"].sum()
        isotopes_list = []
        element, isotopes = self.find_isotopes(display_ion)
        if isotopes.empty:
            message = "1"
        else:
            message = "0"
            isotopes_list = list(isotopes['PeakFunc'])
        return message, frequency_range, peak_sum, isotopes_list


class IID_MainWindow(QMainWindow):
    '''
    The IID GUI Window
    '''
    def __init__(self):
        super().__init__()

        self.title = "Ion Identification Interface"
        self.left = 100
        self.top = 30
        self.width = 760
        self.height = 1000

        # the default parameter setting
        self.directory = "./"
        self.cen_freq = 242.5 # MHz
        self.span = 1000 # kHz

        # the default color setting
        self.bgcolor = "#FAFAFA"
        self.fgcolor = "#23373B"
        self.green = (27,129,62) #"#1B813E"
        self.orange = (233,139,42) #"#E98B2A"
        self.red = (171,59,58) #"#AB3B3A"

        # style for the parameter input bar
        self.input_style = """
        QLineEdit {{
            color: {:s};
            }}
        """.format(self.fgcolor)
        self.lock_style = """
        QLineEdit {{
            color: {:s};
            background-color: lightgray
            }}
        """.format(self.fgcolor)

        # set the font of label
        self.fontProc = QFont("RobotoCondensed", 14)
        self.fontLab = QFont("Inconsolata-dz", 12)        

        self.initUI()

    def QLineEdit_InputStyle(self, lineEdit):
        lineEdit.setStyleSheet(self.input_style)
        lineEdit.setReadOnly(False)

    def QLineEdit_LockStyle(self, lineEdit):
        lineEdit.setStyleSheet(self.lock_style)
        lineEdit.setReadOnly(True)

    def QTable_setModel(self, pandasData, qtable):
        model = PandasModel(pandasData)
        qtable.setModel(model)

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setStyleSheet("QLabel{{color: {0:s} }} QCheckBox{{background-color: {1:s}; color: {0:s}}} QTextEdit{{color: {0:s}}} QMainWindow{{ background-color: {1:s} }} QCentralWidget{{ background-color: {1:s} }} QGroupBox{{ background-color: {1:s} }}".format(self.fgcolor, self.bgcolor))
        pg.setConfigOptions(background=self.bgcolor, foreground=self.fgcolor, antialias=True, imageAxisOrder="row-major")

        self.threadPool = QThreadPool()

        self.setDisplayPanel()
        self.buildConnection()        

        self.statusBar().showMessage("Check the 'File Load' before loading file")

    def setDisplayPanel(self):
        # file list
        self.mFileList = QFileSystemModel(self)
        self.mFileList.setFilter(QDir.Files)
        self.mFileList.setNameFilters(["*.lpp"])
        self.mFileList.setNameFilterDisables(False) 
        self.mFileList.sort(3, Qt.DescendingOrder) # sort by the fourth column, i.e. modified time
        self.vFileList = QTreeView(self)
        self.vFileList.setModel(self.mFileList)
        self.vFileList.setRootIndex(self.mFileList.setRootPath(self.directory))
        self.vFileList.setFont(self.fontLab)
        self.vFileList.header().setStyleSheet("QTreeView { font-size: 12 px;}")
        self.vFileList.hideColumn(1)
        self.vFileList.hideColumn(2)
        self.vFileList.hideColumn(3)

        # set the center frequency and span input
        self.cenFreqLab = QLabel("center frequency [MHz]")
        self.cenFreqLab.setFont(self.fontLab)
        self.cenFreqInput = QLineEdit(str(self.cen_freq))
        self.cenFreqInput.setFont(self.fontProc)
        self.QLineEdit_InputStyle(self.cenFreqInput)

        self.spanLab = QLabel("span [kHz]")
        self.spanLab.setFont(self.fontLab)
        self.spanInput = QLineEdit(str(self.span))
        self.spanInput.setFont(self.fontProc)
        self.QLineEdit_InputStyle(self.spanInput)

        self.FileCheck = QCheckBox("File Load", self)
        self.FileCheck.setFont(self.fontLab)
        self.FileCheck.setStyleSheet("QCheckBox::indicator::unchecked{border: 1px groove silver; background-color:white}")

        # set the ion check box
        self.IonCheck = QCheckBox("Element", self)
        self.IonCheck.setFont(self.fontLab)
        self.IonCheck.setStyleSheet("QCheckBox::indicator::unchecked{border: 1px groove silver; background-color:white}")
        self.IonInput = QLineEdit("", self)
        self.IonInput.setFont(self.fontProc)
        self.QLineEdit_LockStyle(self.IonInput) 
        self.IonButton = QPushButton("Search", self)
        self.IonButton.setFont(self.fontLab)

        # calibrate the ion
        self.CalibrateIonLab = QLabel("Ion (e.g. 3H2)", self)
        self.CalibrateIonLab.setFont(self.fontLab)
        self.CalibrateIonInput = QLineEdit("", self)
        self.CalibrateIonInput.setFont(self.fontProc)
        self.CalibrateBrhoCheck = QCheckBox("Brho [Tm] (e.g. 7.098)", self)
        self.CalibrateBrhoCheck.setFont(self.fontLab)
        self.CalibrateBrhoCheck.setStyleSheet("QCheckBox::indicator::unchecked{border: 1px groove silver; background-color:white}")
        self.CalibrateBrhoInput = QLineEdit("", self)
        self.CalibrateBrhoInput.setFont(self.fontProc)
        self.QLineEdit_LockStyle(self.CalibrateBrhoInput)
        self.CalibratePeakLab = QLabel("PeakLoc [kHz] (e.g. 113)")
        self.CalibratePeakLab.setFont(self.fontLab)
        self.CalibratePeakInput = QLineEdit("",self)
        self.CalibratePeakInput.setFont(self.fontProc)
        self.CalibrateHarmLab = QLabel("Harmonic (e.g. 151)")
        self.CalibrateHarmLab.setFont(self.fontLab)
        self.CalibrateHarmInput = QLineEdit("",self)
        self.CalibrateHarmInput.setFont(self.fontProc)
        self.CalibrateButton = QPushButton("Calibrate", self)
        self.CalibrateButton.setFont(self.fontLab)

        # ion information list
        self.IonTable = QTableView(self)
        self.IonTable.setSortingEnabled(True)
        self.QTable_setModel(pd.DataFrame(columns=['Ion', 'Yield', 'HalfLife', 'Harmonic', 'PeakLoc', 'RevFreq', 'Weight']), self.IonTable)

        # plot spectrum
        self.win = pg.GraphicsLayoutWidget()
        self.spectrum = self.win.addPlot(title="Simulation of the Schottky spectrum")
        self.spectrum.setLogMode(False,True)

        # marker of the ions in spectrum
        self.crosshair_v = pg.InfiniteLine(pos=0, angle=90, pen=self.fgcolor)
        self.spectrum.addItem(self.crosshair_v, ignoreBounds=True)

        # set the Ion panel
        IonGrid = QGridLayout()
        IonGrid.setSpacing(5)
        IonGrid.addWidget(self.vFileList, 0, 0, 6, 3)
        IonGrid.addWidget(self.FileCheck, 0, 3, 1, 1)
        IonGrid.addWidget(self.CalibrateButton, 0, 4, 1, 1)
        IonGrid.addWidget(self.cenFreqLab, 1, 3, 1, 1)
        IonGrid.addWidget(self.cenFreqInput, 2, 3, 1, 1)
        IonGrid.addWidget(self.spanLab, 1, 4, 1, 1)
        IonGrid.addWidget(self.spanInput, 2, 4, 1, 1)
        IonGrid.addWidget(self.CalibrateIonLab, 3, 3, 1, 1)
        IonGrid.addWidget(self.CalibrateIonInput, 4, 3, 1, 1)#, Qt.AlignHCenter)
        IonGrid.addWidget(self.CalibrateBrhoCheck, 3, 4, 1, 1)
        IonGrid.addWidget(self.CalibrateBrhoInput, 4, 4, 1, 1)
        IonGrid.addWidget(self.CalibratePeakLab, 5, 3, 1, 1)
        IonGrid.addWidget(self.CalibratePeakInput, 6, 3, 1, 1)
        IonGrid.addWidget(self.CalibrateHarmLab, 5, 4, 1, 1)
        IonGrid.addWidget(self.CalibrateHarmInput, 6, 4, 1, 1)
        IonGrid.addWidget(self.IonCheck, 6, 0, 1, 1)#, Qt.AlignHCenter)
        IonGrid.addWidget(self.IonInput, 6, 1, 1, 1)
        IonGrid.addWidget(self.IonButton, 6, 2, 1, 1)
        self.IonPanel = QGroupBox()
        self.IonPanel.setStyleSheet("QGroupBox{border: 1px groove silver; margin: 1px; padding-top: 0}")
        self.IonPanel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.IonPanel.setLayout(IonGrid)        

        # set layout
        windowLayout = QVBoxLayout()
        windowLayout.setSpacing(5)
        windowLayout.addWidget(self.win)
        windowLayout.addWidget(self.IonTable)
        windowLayout.addWidget(self.IonPanel)
        wid = QWidget()
        self.setCentralWidget(wid)
        wid.setLayout(windowLayout)


    def buildConnection(self):
        # build connection with element input - checkbox
        def check_elementInput():
            if self.IonCheck.isChecked():
                self.QLineEdit_InputStyle(self.IonInput)
            else:
                self.QLineEdit_LockStyle(self.IonInput)
        self.IonCheck.stateChanged.connect(check_elementInput)

        def check_BrhoInput():
            if self.CalibrateBrhoCheck.isChecked():
                self.QLineEdit_InputStyle(self.CalibrateBrhoInput)
                self.QLineEdit_LockStyle(self.CalibratePeakInput)
                self.QLineEdit_LockStyle(self.CalibrateHarmInput)
            else:
                self.QLineEdit_LockStyle(self.CalibrateBrhoInput)
                self.QLineEdit_InputStyle(self.CalibratePeakInput)
                self.QLineEdit_InputStyle(self.CalibrateHarmInput)
        self.CalibrateBrhoCheck.stateChanged.connect(check_BrhoInput)

        def button_ionCalibrate():
            if self.CalibrateBrhoCheck.isChecked():
                if self.CalibrateBrhoInput.text() == "":
                    self.statusBar().showMessage("No valid input!")
                    return
                else:
                    self.statusBar().showMessage("Calibrating ...")
            else:
                if self.CalibrateIonInput.text() == "" or self.CalibratePeakInput.text() == "" or self.CalibrateHarmInput.text()== "":
                    self.statusBar().showMessage("No valid input!")
                    return
                if self.CalibrateIonInput.text() in self.peak_list['Ion'].values:
                    self.statusBar().showMessage("Calibrating ...")
                else:
                    self.statusBar().showMessage("no valid ion input!")
                    return 

            def data_calibrate_worker():
                if self.CalibrateBrhoCheck.isChecked():
                    self.peak_list = self.FileWork.calibrate_Brho(float(self.CalibrateBrhoInput.text()))
                else:
                    self.peak_list = self.FileWork.calibrate_peak_loc(self.CalibrateIonInput.text(), float(self.CalibratePeakInput.text()), int(self.CalibrateHarmInput.text()))
                self.ion = re.sub("[^A-Za-z]", "", self.peak_list["Ion"][0])
                self.message, self.frequency_range, self.peak_sum, self.isotopes_list = self.FileWork.gaussian_peak(self.ion)

            def data_calibrate_ready():
                self.spectrum.clear()
                self.spectrum.plot(self.frequency_range, self.peak_sum, pen=pg.mkPen(self.green, width=2))
                for isotope in self.isotopes_list:
                    self.spectrum.plot(self.frequency_range, isotope, pen=pg.mkPen(self.orange, width=2))
                    #self.spectrum.showGrid(x=True, y=True)
                self.spectrum.addItem(self.crosshair_v, ignoreBounds=True)
                self.crosshair_v.setValue(0)
                self.QTable_setModel(self.peak_list, self.IonTable)
                self.IonInput.setText(self.ion)
                self.statusBar().showMessage("Data has been calibrated!")
                if not self.CalibrateBrhoCheck.isChecked():
                    self.CalibratePeakInput.setText("{:.0f}".format(self.peak_list[self.peak_list['Ion'].isin([self.CalibrateIonInput.text()])]["PeakLoc"].values[0]))
                    self.CalibrateHarmInput.setText("{:d}".format(self.peak_list[self.peak_list['Ion'].isin([self.CalibrateIonInput.text()])]["Harmonic"].values[0]))
                self.CalibrateBrhoInput.setText("{:.5f}".format(self.FileWork.Brho))
        
            worker = Worker(data_calibrate_worker)
            worker.signals.finished.connect(data_calibrate_ready)
            self.threadPool.start(worker) 
        self.CalibrateButton.clicked.connect(button_ionCalibrate)

        def button_elementSearch():
            if self.IonCheck.isChecked():
                if self.IonInput.text() == "":
                    self.statusBar().showMessage("No ion input! Show the whole ion list instead.")

                def data_flash_worker():
                    self.ion, self.peak_list_f = self.FileWork.find_isotopes(self.IonInput.text())
                    self.message, self.frequency_range, self.peak_sum, self.isotopes_list = self.FileWork.gaussian_peak(self.ion)
                def data_flash_ready():
                    self.spectrum.clear()
                    self.spectrum.plot(self.frequency_range, self.peak_sum, pen=pg.mkPen(color=self.green, width=2))
                    try:
                        for isotope in self.isotopes_list:
                            self.spectrum.plot(self.frequency_range, isotope, pen=pg.mkPen(color=self.orange, width=2))
                    except:
                        pass
                    self.spectrum.addItem(self.crosshair_v, ignoreBounds=True)
                    self.crosshair_v.setValue(0)
                    if self.ion == "":
                        self.statusBar().showMessage("No valid ion input!")
                    try:
                        self.QTable_setModel(self.peak_list_f.drop(columns='PeakFunc'), self.IonTable)
                    except:
                        self.QTable_setModel(self.peak_list_f, self.IonTable)
                    self.IonInput.setText(self.ion)
                worker = Worker(data_flash_worker)
                worker.signals.finished.connect(data_flash_ready)
                self.threadPool.start(worker)

            else:
                self.statusBar().showMessage("Check the 'Element' first!")
        self.IonButton.clicked.connect(button_elementSearch)

        def check_fileLoad():
            if self.FileCheck.isChecked():
                self.QLineEdit_LockStyle(self.cenFreqInput)
                self.QLineEdit_LockStyle(self.spanInput)
                self.vFileList.activated.connect(selected_file)
            else:
                self.QLineEdit_InputStyle(self.cenFreqInput)
                self.QLineEdit_InputStyle(self.spanInput)
                self.vFileList.activated.disconnect(selected_file)
        self.FileCheck.stateChanged.connect(check_fileLoad)

        def selected_file(model_index):
            self.load_file(model_index.data())            

        def on_moved_spectrum(point):
            if self.spectrum.sceneBoundingRect().contains(point):
                coords = self.spectrum.getViewBox().mapSceneToView(point)
                self.crosshair_v.setValue(coords.x())
                try:
                    ions = self.FileWork.find_ion(coords.x())
                    self.statusBar().showMessage("δf = {:.5g} kHz, ion = {:6s}, half = {:11s}".format(coords.x(), ions["Ion"][0], ions["HalfLife"][0]))
                except:    
                    self.statusBar().showMessage("δf = {:.5g} kHz".format(coords.x()))
        self.spectrum.scene().sigMouseMoved.connect(on_moved_spectrum)
                
    def load_file(self, data_file):
        self.statusBar().showMessage("file loading...")
        self.cen_freq = float(self.cenFreqInput.text())
        self.span = float(self.spanInput.text())

        def load_file_worker(data_file):
            self.FileWork = IID(data_file, self.cen_freq, self.span)
            self.peak_list = self.FileWork.calc_peak()
            self.ion = re.sub("[^A-Za-z]", "", self.peak_list["Ion"][0])
            self.message, self.frequency_range, self.peak_sum, self.isotopes_list = self.FileWork.gaussian_peak(self.ion)
    
        def load_file_ready():
            self.spectrum.clear()
            self.spectrum.plot(self.frequency_range, self.peak_sum, pen=pg.mkPen(self.green, width=2))
            for isotope in self.isotopes_list:
                self.spectrum.plot(self.frequency_range, isotope, pen=pg.mkPen(self.orange, width=2))
            #self.spectrum.showGrid(x=True, y=True)
            self.spectrum.addItem(self.crosshair_v, ignoreBounds=True)
            self.crosshair_v.setValue(0)
            self.QTable_setModel(self.peak_list, self.IonTable)
            self.CalibrateBrhoInput.setText("{:.5f}".format(self.FileWork.Brho))
            self.IonInput.setText(self.ion)
            self.statusBar().showMessage("Selected file has been loaded!")
        
        worker = Worker(load_file_worker, data_file)
        worker.signals.finished.connect(load_file_ready)
        self.threadPool.start(worker) 
    

    def keyPressEvent(self, event):
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_W:
            reply = QMessageBox.question(self, "Message", "Are you sure to quit?", QMessageBox.Yes|QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                sys.exit()
            else:
                return


if __name__ == '__main__':
    app = QApplication(sys.argv)
    iid_app = IID_MainWindow()

    iid_app.show()
    sys.exit(app.exec())
