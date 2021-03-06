#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys, re, io, time
import numpy as np
import pandas as pd
import pyqtgraph as pg

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.uic import *

from ion_id import IID
from pandas_model import PandasModel
from multithread import Worker

class IID_GUI(QMainWindow):
    '''
    A GUI for showing the ion identification result
    '''
    bgcolor = "#FAFAFA"
    fgcolor = "#23373B"
    green = (27, 129, 62)   # "#1B813E"
    orange = (233, 139, 42) # "#E98B2A"
    red = (171, 59, 58)     # "#AB3B3A"
    
    def __init__(self):
        super().__init__()
        
        # the default parameter setting
        self.directory = "./"
        self.cen_freq = 242.5 # MHz
        self.span = 1000 # kHz

        self.set_display()
        self.build_connections()
        self.threadPool = QThreadPool()

        self.statusBar().showMessage("Select the .lpp file, and check the 'File Load' before loading file")


    def QLineEdit_InputStyle(self, lineEdit):
        lineEdit.setStyleSheet("QLineEdit {{color : {:s}}}".format(self.fgcolor))
        lineEdit.setReadOnly(False)

    def QLineEdit_LockStyle(self, lineEdit):
        lineEdit.setStyleSheet("QLineEdit {{color: {:s}; background-color: lightgray}}".format(self.fgcolor))
        lineEdit.setReadOnly(True)

    def QCheckBox_InputStyle(self, checkBox):
        checkBox.setStyleSheet("QCheckBox {{color: {:s}}}".format(self.fgcolor))
        checkBox.setEnabled(True)

    def QCheckBox_LockStyle(self, checkBox):
        checkBox.setStyleSheet("QCheckBox {{color: {:s}; background-color: lightgray}}".format(self.fgcolor))
        checkBox.setEnabled(False)
        checkBox.setChecked(False)
        
    def QTable_setModel(self, pandasData, qtable):
        model = PandasModel(pandasData)
        qtable.setModel(model)

    def spectrum_font_label(self, labs): return "<span style=font-family:Roboto;font-size=14pt>" + labs + "</span>"

    def set_display(self):
        pg.setConfigOptions(background=self.bgcolor, foreground=self.fgcolor, antialias=True, imageAxisOrder="row-major")
        loadUi("iid_GUI.ui", self) # the ui. file is generated by the Qt designer

        # set parameter
        self.cenFreqInput.setText(str(self.cen_freq))
        self.spanInput.setText(str(self.span))
        self.QLineEdit_InputStyle(self.cenFreqInput)
        self.QLineEdit_InputStyle(self.spanInput)
        self.QLineEdit_LockStyle(self.ionInput)
        self.QLineEdit_LockStyle(self.calibrateBrhoInput)
        self.QCheckBox_LockStyle(self.bareCheck)
        self.QCheckBox_LockStyle(self.HLikeCheck)
        self.QCheckBox_LockStyle(self.HeLikeCheck)

        # ion spectrum
        self.spectrum = self.gSpectrum.addPlot(title=self.spectrum_font_label("Simulation of the Schottky spectrum"))
        self.spectrum.setLogMode(False, True)
        self.spectrum.setLabels(left=self.spectrum_font_label("Magnitude"), bottom=self.spectrum_font_label("Frequency - ___ MHz [kHz]"))
        # marker of the ions in spectrum
        self.crosshair_v = pg.InfiniteLine(pos=0, angle=90, pen=self.fgcolor)
        self.spectrum.addItem(self.crosshair_v, ignoreBounds=True)

        # ion information list
        self.ionTable.setSortingEnabled(True)
        self.QTable_setModel(pd.DataFrame(columns=["Ion", "Yield", "Half-Life", "Harmonic", "PeakLoc", "RevFreq", "Weight"]), self.ionTable)

        # file list
        self.mFileList = QFileSystemModel(self)
        self.mFileList.setFilter(QDir.Files)
        self.mFileList.setNameFilters(["*.lpp"])
        self.mFileList.setNameFilterDisables(False)
        self.mFileList.sort(3, Qt.DescendingOrder) # sort by the fourth column, i.e. modified time
        self.fileList.setModel(self.mFileList)
        self.fileList.setRootIndex(self.mFileList.setRootPath(self.directory))
        self.fileList.hideColumn(1)
        self.fileList.hideColumn(2)
        self.fileList.hideColumn(3)
    
    def build_connections(self):
        # build connection with calibration
        def check_BrhoInput():
            if self.calibrateBrhoCheck.isChecked():
                self.QLineEdit_InputStyle(self.calibrateBrhoInput)
                self.QLineEdit_LockStyle(self.calibratePeakLocInput)
                self.QLineEdit_LockStyle(self.calibrateHarmInput)
                self.QLineEdit_LockStyle(self.calibrateIonInput)
            else:
                self.QLineEdit_LockStyle(self.calibrateBrhoInput)
                self.QLineEdit_InputStyle(self.calibratePeakLocInput)
                self.QLineEdit_InputStyle(self.calibrateHarmInput)
                self.QLineEdit_InputStyle(self.calibrateIonInput)
        self.calibrateBrhoCheck.stateChanged.connect(check_BrhoInput)

        def button_ionCalibrate():
            if self.calibrateBrhoCheck.isChecked():
                if self.calibrateBrhoInput.text() == "":
                    self.statusBar().showMessage("Invalid input!")
                    return
                else:
                    self.statusBar().showMessage("Calibrating...")
            else:
                if self.calibrateIonInput.text() == "" or self.calibratePeakLocInput.text() == "" or self.calibrateHarmInput.text() == "":
                    self.statusBar().showMessage("Invalid input!")
                    return
                if self.calibrateIonInput.text() in self.peak_list["Ion"].values:
                    self.statusBar().showMessage("Calibrating...")
                else:
                    self.statusBar().showMessage("Invalid input!")
                    return

            def data_calibrate_worker():
                if self.calibrateBrhoCheck.isChecked():
                    self.frequency_range, self.peak_sum, self.peak_list = self.FileWork.calibrate_Brho(float(self.calibrateBrhoInput.text()))
                else:
                    self.frequency_range, self.peak_sum, self.peak_list = self.FileWork.calibrate_peak_loc(self.calibrateIonInput.text(), float(self.calibratePeakLocInput.text()), int(self.calibrateHarmInput.text()))
                    
                self.ion = self.peak_list["Ion"][0]#[self.peak_list["Weight"].idxmax()]
                _, self.peak_select_ions = self.calc_selected_ions(self.ion)
        
            def data_calibrate_ready():
                self.display(self.peak_list, self.peak_select_ions)
                self.statusBar().showMessage("Data has been calibrated!")
                if self.calibrateBrhoCheck.isChecked():
                    if self.calibrateIonInput.text() in self.peak_list["Ion"].values:
                        self.calibratePeakLocInput.setText("{:.0f}".format(self.peak_list[self.peak_list["Ion"].isin([self.calibrateIonInput.text()])]["PeakLoc"].values[0]))
                        self.calibrateHarmInput.setText("{:.0f}".format(self.peak_list[self.peak_list["Ion"].isin([self.calibrateIonInput.text()])]["Harmonic"].values[0]))
                    else:
                        self.calibrateIonInput.setText("")
                        self.calibratePeakLocInput.setText("")
                        self.calibrateHarmInput.setText("")
                self.calibrateBrhoInput.setText("{:.5f}".format(self.FileWork.Brho))

            worker = Worker(data_calibrate_worker)
            worker.signals.finished.connect(data_calibrate_ready)
            self.threadPool.start(worker)
        self.calibrateButton.clicked.connect(button_ionCalibrate)

        # build connection with ion search 
        def check_ionInput():
            if self.ionCheck.isChecked():
                self.QLineEdit_InputStyle(self.ionInput)
            else:
                self.QLineEdit_LockStyle(self.ionInput)
        self.ionCheck.stateChanged.connect(check_ionInput)

        def button_ionSearch():
            if self.ionCheck.isChecked():
                if self.ionInput.text() == "":
                    self.statusBar().showMessage("No ion input for search! show the whole ion list instead.")
                    self.ionInput.setText(self.ion)

                def data_flash_worker():
                    self.select_ions, self.peak_select_ions = self.calc_selected_ions(self.ionInput.text())
        
                def data_flash_ready():
                    self.display(self.select_ions, self.peak_select_ions)
                    self.statusBar().showMessage("Ion founded!")

                worker = Worker(data_flash_worker)
                worker.signals.finished.connect(data_flash_ready)
                self.threadPool.start(worker)
            else:
                self.statusBar().showMessage("Check the 'Ion' first!")
        self.ionButton.clicked.connect(button_ionSearch)

        # build connection with ion-types (bare, H-like, He-like)
        def check_ionType():
            self.display_ions = pd.DataFrame(columns=["Ion", "Yield", "Half-Life", "Harmonic", "PeakLoc", "RevFreq", "Weight"])
            if self.bareCheck.isChecked():
                self.display_ions = pd.concat([self.display_ions, self.bare_ions])
            if self.HLikeCheck.isChecked():
                self.display_ions = pd.concat([self.display_ions, self.H_like_ions])
            if self.HeLikeCheck.isChecked():
                self.display_ions = pd.concat([self.display_ions, self.He_like_ions])
            if len(self.display_ions) != 0:
                _, self.peak_select_ions = self.calc_selected_ions(self.display_ions["Ion"][self.display_ions["Weight"].idxmax()])
                self.display(self.display_ions, self.peak_select_ions)
        self.bareCheck.stateChanged.connect(check_ionType)
        self.HLikeCheck.stateChanged.connect(check_ionType)
        self.HeLikeCheck.stateChanged.connect(check_ionType)

        # build connection with file load
        def check_fileLoad():
            if self.fileCheck.isChecked():
                self.QLineEdit_LockStyle(self.cenFreqInput)
                self.QLineEdit_LockStyle(self.spanInput)
                self.bareCheck.stateChanged.connect(check_ionType)
                self.HLikeCheck.stateChanged.connect(check_ionType)
                self.HeLikeCheck.stateChanged.connect(check_ionType)
                self.QCheckBox_LockStyle(self.bareCheck)
                self.QCheckBox_LockStyle(self.HLikeCheck)
                self.QCheckBox_LockStyle(self.HeLikeCheck)
                self.fileList.activated.connect(selected_file)
            else:
                self.QLineEdit_InputStyle(self.cenFreqInput)
                self.QLineEdit_InputStyle(self.spanInput)
                self.bareCheck.stateChanged.disconnect(check_ionType)
                self.HLikeCheck.stateChanged.disconnect(check_ionType)
                self.HeLikeCheck.stateChanged.disconnect(check_ionType)
                self.QCheckBox_InputStyle(self.bareCheck)
                self.QCheckBox_InputStyle(self.HLikeCheck)
                self.QCheckBox_InputStyle(self.HeLikeCheck)
                self.fileList.activated.disconnect(selected_file)
        self.fileCheck.stateChanged.connect(check_fileLoad)

        def selected_file(model_index):
            self.load_file(model_index.data())

        def on_moved_spectrum(point):
            if self.spectrum.sceneBoundingRect().contains(point):
                coords = self.spectrum.getViewBox().mapSceneToView(point)
                self.crosshair_v.setValue(coords.x())
                try:
                    ion_ind = self.find_ions(coords.x())
                    self.statusBar().showMessage("δf = {:.5g} kHz, ion = {:6s}".format(coords.x(), self.peak_list["Ion"][ion_ind]))
                except:
                    self.statusBar().showMessage("δf = {:.5g} kHz".format(coords.x()))
        self.spectrum.scene().sigMouseMoved.connect(on_moved_spectrum)
                    
            
    def load_file(self, data_file):
        self.QCheckBox_LockStyle(self.bareCheck)
        self.QCheckBox_LockStyle(self.HLikeCheck)
        self.QCheckBox_LockStyle(self.HeLikeCheck)
        self.statusBar().showMessage("file loading...")
        self.cen_freq = float(self.cenFreqInput.text())
        self.span = float(self.spanInput.text())

        def load_file_worker(data_file):
            self.FileWork = IID(data_file, self.cen_freq, self.span, n_peak=10, GUI_mode=True)
            self.frequency_range, self.peak_sum, self.peak_list = self.FileWork.calc_peak()
            self.ion = self.peak_list["Ion"][0]
        
        def load_file_ready():
            # check the bare ions
            self.bare_ions = self.peak_list[self.peak_list["Half-Life"].str.match("([\w.]+ ?[\w.]+)$")]
            self.QCheckBox_InputStyle(self.bareCheck)
            self.bareCheck.setChecked(True)
            # check the H-like ions
            if True in self.peak_list["Half-Life"].str.match("([\w.]+ ?[\w.]+) [\*]$").values:
                self.H_like_ions = self.peak_list[self.peak_list["Half-Life"].str.match("([\w.]+ ?[\w.]+) [\*]$")]
                self.QCheckBox_InputStyle(self.HLikeCheck)
                self.HLikeCheck.setChecked(True)
            # check the He-like ions
            if True in self.peak_list["Half-Life"].str.match("([\w.]+ ?[\w.]+) [\*]{2}$").values:
                self.He_like_ions = self.peak_list[self.peak_list["Half-Life"].str.match("([\w.]+ ?[\w.]+) [\*]{2}$")]
                self.QCheckBox_InputStyle(self.HeLikeCheck)
                self.HeLikeCheck.setChecked(True)
            self.statusBar().showMessage("Selected file has been loaded!")

        worker = Worker(load_file_worker, data_file)
        worker.signals.finished.connect(load_file_ready)
        self.threadPool.start(worker)
    
    def calc_selected_ions(self, ion):
        '''
        show the information of the selected element
        '''
        if ion.isdigit() and (True in self.display_ions['Ion'].str.match(ion+"([A-Za-z]+)(\d+)").values): # match A, e.g. 3
            ind = self.display_ions.index[self.display_ions['Ion'].str.match(ion+"([A-Za-z]+)(\d+)")].tolist()
        elif (bool(re.fullmatch("[A-Z]", ion)) or bool(re.fullmatch("[A-Z][a-z]", ion))) and (True in self.display_ions['Ion'].str.match("(\d+)"+ion+"(\d+)").values): # match element, e.g. H
            ind = self.display_ions.index[self.display_ions['Ion'].str.match("(\d+)"+ion+"(\d+)")].tolist()
        elif ("+" in ion) and ion[:-1].isdigit() and (True in self.display_ions['Ion'].str.match("(\d+)([A-Za-z]+)"+ion[:-1]).values): # match charge, e.g. 1+
            ind = self.display_ions.index[self.display_ions['Ion'].str.match("(\d+)([A-Za-z]+)"+ion[:-1])].tolist()
        elif bool(re.fullmatch("(\d+)[A-Za-z]+(\d+)", ion)) and (ion in self.display_ions['Ion'].values): # match AElementQ, e.g. 3H1
            ind = self.display_ions.index[self.display_ions['Ion']==ion].tolist()
        elif bool(re.fullmatch("(\d+)[A-Za-z]+", ion)) and (True in self.display_ions['Ion'].str.match(ion+"(\d+)").values): # match AElement, e.g. 3H
            ind = self.display_ions.index[self.display_ions['Ion'].str.match(ion+"(\d+)")].tolist()
        elif (bool(re.fullmatch("[A-Z]\d+", ion)) or bool(re.fullmatch("[A-Z][a-z]\d+", ion))) and (True in self.display_ions['Ion'].str.match("(\d+)"+ion).values): # match ElementQ, e.g. H1
            ind = self.display_ions.index[self.display_ions['Ion'].str.match("(\d+)"+ion)].tolist()
        else:
            ion = self.ion
            ind = self.peak_list.index[self.peak_list["Ion"]==ion].tolist()
            self.ionInput.setText(ion)
            self.statusBar().showMessage("Invalid ion input! Change to the default instead.")
        select_ions = self.peak_list.loc[ind, :]
        lim = self.display_ions["Weight"].max() / 1e5
        def calc_ion_peak(x):
            width = self.FileWork.sigma * x["Harmonic"] * x["RevFreq"] * 1e3 / 1.66
            a = x["Weight"] / (np.sqrt(2 * np.pi) * width) * np.exp(-(self.frequency_range - x["PeakLoc"])**2 / (2 * width**2))
            a[a < lim] = lim
            return a
        peak_select_ions = select_ions.apply(lambda x: calc_ion_peak(x), axis=1)
        return select_ions.loc[:, ["Ion", "Yield", "Half-Life", "Harmonic", "PeakLoc", "RevFreq", "Weight"]], peak_select_ions

    def display(self, ion_table, emphasized_ions):
        '''
        flash the result of GUI display, including the ionTable and spectrum
        '''
        self.spectrum.clear()
        self.spectrum.plot(self.frequency_range, self.peak_sum, pen=pg.mkPen(self.green, width=2))
        if len(emphasized_ions) != 0:
            for ion_peak in emphasized_ions:
                self.spectrum.plot(self.frequency_range, ion_peak, pen=pg.mkPen(self.orange, width=2))
        self.spectrum.setLabels(left=self.spectrum_font_label("Magnitude"), bottom=self.spectrum_font_label("Frequency - {:g} MHz [kHz]".format(self.cen_freq)))
        self.spectrum.setRange(xRange=(-self.span/2,self.span/2))
        self.spectrum.addItem(self.crosshair_v, ignoreBounds=True)
        self.crosshair_v.setValue(ion_table["PeakLoc"][ion_table["Weight"].idxmax()])
        self.QTable_setModel(ion_table, self.ionTable)
        self.calibrateBrhoInput.setText("{:.5f}".format(self.FileWork.Brho))
        self.ionInput.setText(ion_table["Ion"][ion_table["Weight"].idxmax()])


    def find_ions(self, coord_x):
        '''
        show the information of the selected ion peak
        '''
        if self.peak_list[np.abs(self.peak_list["PeakLoc"] - coord_x) <= (self.FileWork.sigma * self.peak_list["Harmonic"].max() * self.peak_list["RevFreq"].max()*1e3)]["Ion"].count() > 0:
            return np.abs(self.peak_list["PeakLoc"] - coord_x).idxmin()

    def keyPressEvent(self, event):
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_W:
            reply = QMessageBox.question(self, "Message", "Are you sure to quit?", QMessageBox.Yes|QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                sys.exit()
            else:
                return


if __name__ == '__main__':
    app = QApplication(sys.argv)
    iid_app = IID_GUI()

    iid_app.show()
    sys.exit(app.exec())
