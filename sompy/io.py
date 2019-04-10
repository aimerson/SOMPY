#! /usr/bin/env python

__author__ = ("Alex Merson")
__version__ = "1.0.0"

import sys,fnmatch
import numpy as np
from .hdf5 import HDF5
from .codebook import Codebook
from .sompy import SOM

    

class OutputToHDF5(HDF5):
    
    def __init__(self,outfile):
        super(self.__class__,self).__init__(outfile,'w')        
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        return
    
    def __call__(self,SOM,verbose=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.writeSOM(SOM,verbose=verbose)
        return

    def writeCodebook(self,codebook,verbose=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if verbose:
            print("Writing Codebook to file...")
        self.mkGroup("/codebook")
        attrib = {}
        attrib["lattice"] = codebook.lattice
        attrib["nnodes"] = codebook.nnodes
        attrib["initialized"] = codebook.initialized
        attrib["mapsize"] = codebook.mapsize
        self.addAttributes("/codebook",attrib)                
        self.writeSOMDataset(codebook.matrix,"matrix",hdfdir="/codebook")
        return

    def writeSOMDataset(self,arr,name,hdfdir="/",verbose=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if verbose:
            print("Writing "+name+" to file...")
        self.addDataset(hdfdir,name,arr,maxshape=arr.shape)
        return

    def writeSOM(self,SOM,storeNormalizedData=False,verbose=False):                
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if verbose:
            print("Writing SOM to file...")
        # Store basic attributes to dictionary
        attrib = {}
        keys = ["name","training","initialization","_dim","_dlabel","_dlen","mapshape"]
        for key in keys:
            if SOM.__dict__[key] is None:
                attrib[key] = "None"
            else:
                attrib[key] = SOM.__dict__[key]
        # Write name of neighborhood and normalizer objects
        attrib['neighborhood'] = SOM.neighborhood.name
        attrib['_normalizer'] = SOM._normalizer.name
        # Write attributes to file
        self.addAttributes("/",attrib)                
        # Write BMU
        self.writeSOMDataset(SOM._bmu,"_bmu",hdfdir="/",verbose=verbose)
        # Write mask
        self.writeSOMDataset(SOM.mask,"mask",hdfdir="/",verbose=verbose)
        # Write _data
        if storeNormalizedData:
            self.writeSOMDataset(SOM._data,"_data",hdfdir="/",verbose=verbose)
        # Write data_raw
        self.writeSOMDataset(SOM.data_raw,"data_raw",hdfdir="/",verbose=verbose)
        # Write distance matrix
        self.writeSOMDataset(SOM._distance_matrix,"_distance_matrix",hdfdir="/",verbose=verbose)
        # Write component names
        self.writeSOMDataset(np.array(SOM._component_names),"_component_names",hdfdir="/",verbose=verbose)
        # Write codebook
        self.writeCodebook(SOM.codebook,verbose=verbose)
        return
    
    
class InputFromHDF5(HDF5):

    def __init__(self,hdf5File):
        super(self.__class__,self).__init__(hdf5File,'r')
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        return

    def __call__(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        return self.createSOM()
        

    def createCodebook(self,verbose=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Create codebook object
        if verbose:
            print("Creating codebook...")
        attrib = self.readAttributes("/codebook")
        CODE = Codebook(list(attrib["mapsize"]),lattice=attrib["lattice"])
        CODE.nnodes = int(attrib["nnodes"])
        CODE.matrix = np.array(self.fileObj["/codebook/matrix"])
        CODE.initialized = attrib["initialized"]
        return CODE        
    
    def createNeighborhood(self,verbose=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if verbose:
            print("Creating neighborhood object...")
        # Get name for neighborhood object
        name = self.readAttributes("/",required=["neighborhood"])["neighborhood"]
        # Create neighborhood object
        if fnmatch.fnmatch(name,"gaussian"):
            from .neighborhood import GaussianNeighborhood
            NEIGHBORHOOD = GaussianNeighborhood()
        elif fnmatch.fnmatch(name,"bubble"):
            from .neighborhood import BubbleNeighborhood
            NEIGHBORHOOD = BubbleNeighborhood()
        else:
            raise ValueError("ERROR! "+funcname+"(): Neighborhood not recognized! "+\
                                 "Should be 'gaussian' or 'bubble'.")
        return NEIGHBORHOOD
    
    
    def createNormalizer(self,verbose=False):        
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if verbose:
            print("Creating normalizer object...")
        # Get name for normalizer
        name = self.readAttributes("/",required=["_normalizer"])["_normalizer"]
        # Create normalizer object
        if fnmatch.fnmatch(name,"var"):
            from .normalization import VarianceNormalizator
            NORM = VarianceNormalizator()
        elif fnmatch.fnmatch(name,"range"):
            from .normalization import RangeNormalizator
            NORM = RangeNormalizator()
        elif fnmatch.fnmatch(name,"log"):
            from .normalization import LogNormalizator
            NORM = LogNormalizator()
        elif fnmatch.fnmatch(name,"logistic"):
            from .normalization import LogisticNormalizator
            NORM = LogisticNormalizator()
        elif fnmatch.fnmatch(name,"histd"):
            from .normalization import HistDNormalizator
            NORM = HistDNormalizator()
        elif fnmatch.fnmatch(name,"histc"):
            from .normalization import HistCNormalizator
            NORM = HistCNormalizator()
        else:
            raise ValueError("ERROR! "+funcname+"(): Normalization not recognized! "+\
                                 "Should be 'var', 'range', 'log', 'logistic',"+\
                                 " 'histd' or 'histc'.")
        return NORM
        
            
    def createSOM(self,verbose=False):        
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if verbose:
            print("Creating SOM from file...")
        # Read and create codebook object
        CODE = self.createCodebook(verbose=verbose)
        # Extract raw data
        data = np.array(self.fileObj["/data_raw"])
        # Extract component names
        components = np.array(self.fileObj["/_component_names"])        
        # Create neighborhood and normalizer classes
        neighborhood = self.createNeighborhood(verbose=verbose)
        normalizer = self.createNormalizer(verbose=verbose)
        # Read remaining attributes
        attrib = self.readAttributes("/")
        # Create SOM object
        som = SOM(data,neighborhood,normalizer=normalizer,\
                      mapsize=CODE.mapsize,mask=None,mapshape=attrib["mapshape"],\
                      lattice=CODE.lattice,initialization=attrib["initialization"],\
                      training=attrib["training"],name=attrib["name"],\
                      component_names=list(components))
        # Update attributes
        if verbose:
            print("Updating SOM attributes...")
        som.mask = np.array(self.fileObj["/mask"])
        som._dlabel = attrib["_dlabel"]
        som._bmu = np.array(self.fileObj["/_bmu"])
        som.codebook = CODE
        som._distance_matrix = np.array(self.fileObj["/_distance_matrix"])
        return som


