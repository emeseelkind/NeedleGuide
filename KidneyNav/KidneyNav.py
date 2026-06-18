import logging
import numpy as np
import os
import json
from datetime import datetime, timezone
from typing import Annotated, Optional
import qt
import vtk
import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import (
    vtkMRMLScalarVolumeNode, 
    vtkMRMLVolumeReconstructionNode, 
    vtkMRMLMarkupsFiducialNode, 
    vtkMRMLModelNode, 
    vtkMRMLLinearTransformNode, 
    vtkMRMLIGTLConnectorNode,
    vtkMRMLSequenceBrowserNode
)


#
# KidneyNav
#


class KidneyNav(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("KidneyNav")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Ultrasound")]
        self.parent.dependencies = ["VolumeResliceDriver"]  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Tamas Ungi (Queen's University)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#KidneyNav">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # KidneyNav1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="KidneyNav",
        sampleName="KidneyNav1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "KidneyNav1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="KidneyNav1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="KidneyNav1",
    )

    # KidneyNav2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="KidneyNav",
        sampleName="KidneyNav2",
        thumbnailFileName=os.path.join(iconsPath, "KidneyNav2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="KidneyNav2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="KidneyNav2",
    )


#
# KidneyNavParameterNode
#


@parameterNodeWrapper
class KidneyNavParameterNode:
    """
    The parameters needed by module.
    """
    inputVolume: vtkMRMLScalarVolumeNode
    referenceToRas: vtkMRMLLinearTransformNode
    imageToReference: vtkMRMLLinearTransformNode
    cadProbeToProbe: vtkMRMLLinearTransformNode
    needleToReference: vtkMRMLLinearTransformNode
    needleTipToneedle: vtkMRMLLinearTransformNode
    needleModel: vtkMRMLModelNode
    predictionToReference: vtkMRMLLinearTransformNode
    predictionVolume: vtkMRMLScalarVolumeNode
    reconstructorNode: vtkMRMLVolumeReconstructionNode
    kidneyNavMarkups: vtkMRMLMarkupsFiducialNode
    plusConnectorNode: vtkMRMLIGTLConnectorNode
    predictionConnectorNode: vtkMRMLIGTLConnectorNode
    blurSigma: Annotated[float, WithinRange(0, 5)] = 0.5
    reconstructedVolume: vtkMRMLScalarVolumeNode
    opacityThreshold: Annotated[int, WithinRange(-100, 200)] = 60
    invertThreshold: bool = False
    showKidney: bool = True
    recordingLeft: bool = True
    recordingRight: bool = False
    inPlaneDisplayMode: str = "Projection"
    sequenceBrowserNode: vtkMRMLSequenceBrowserNode
    checkpointDescription: str = ""

#
# KidneyNavWidget
#

class KidneyNavWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """
    
    LAYOUT_2D3D = 601

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        
        self.displayedReconstructedVolume = None
        self.observedKidneyNavMarkups = None

        # for debugging
        slicer.mymod = self

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/KidneyNav.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = KidneyNavLogic()
        self.logic.setup()

        # Ensure any newly added qMRML widgets receive the MRML scene
        for widgetName in [
            "inputVolumeSelector",
            "predictionVolumeSelector",
            "referenceToRasSelector",
            "probeToReferenceSelector",
            "cadProbeToProbeSelector",
            "reconstructorNodeSelector",
        ]:
            if hasattr(self.ui, widgetName):
                w = getattr(self.ui, widgetName)
                if hasattr(w, "setMRMLScene"):
                    w.setMRMLScene(slicer.mrmlScene)

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartImportEvent, self.onSceneStartImport)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndImportEvent, self.onSceneEndImport)

        # UI widget connections
        self.ui.startOpenIGTLinkButton.connect("toggled(bool)", self.onOpenIGTLinkButton)
        self.ui.applyButton.connect("clicked(bool)", self.onReconstructionButton)
        self.ui.volumeOpacitySlider.connect("valueChanged(int)", self.onVolumeOpacitySlider)
        self.ui.setRoiButton.connect("clicked(bool)", self.onSetRoiButton)
        self.ui.blurButton.connect("clicked()", self.onBlurButton)

        # Explicitly connect signal/slot
        self.ui.inPlaneCheckBox.connect('toggled(bool)', self.onInPlaneOverlayToggled)
        self.ui.outOfPlaneCheckBox.connect('toggled(bool)', self.onOutOfPlaneOverlayToggled)
        if hasattr(self.ui, 'inPlaneDisplayModeComboBox'):
            self.ui.inPlaneDisplayModeComboBox.connect('currentIndexChanged(int)', self.onInPlaneDisplayModeChanged)
        
        # Connect segmentation visualization checkbox (single binary segmentation)
        if hasattr(self.ui, 'showKidneyCheckBox'):
            self.ui.showKidneyCheckBox.connect('toggled(bool)', self.onSegmentationToggled)
        
        # Connect sequence recording controls
        # Kidney side is not used for sequence creation in study workflows.
        # Keep the selector in UI for other workflows, but it does not affect recording.
        self.ui.initializeRecordingButton.connect('clicked(bool)', self.onInitializeRecordingButton)
        self.ui.saveRecordingButton.connect('clicked(bool)', self.onSaveRecordingButton)
        
        # Set default output folder if not already set
        if hasattr(self.ui, 'outputFolderPathLineEdit'):
            if not self.ui.outputFolderPathLineEdit.currentPath:
                defaultPath = os.path.join(os.path.expanduser("~"), "Documents", "KidneyNavRecordings")
                self.ui.outputFolderPathLineEdit.currentPath = defaultPath
       
        # Add custom layout
        self.addCustomLayouts()
        slicer.app.layoutManager().setLayout(self.LAYOUT_2D3D)
        slicer.app.layoutManager().sliceWidget("Red").sliceController().setSliceVisible(True)
        for viewNode in slicer.util.getNodesByClass("vtkMRMLAbstractViewNode"):
            viewNode.SetOrientationMarkerType(slicer.vtkMRMLAbstractViewNode.OrientationMarkerTypeHuman)
        
        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()
        
        # Autofill known scene nodes but keep manual selectors available.
        self.autoFillKnownSceneNodes()
        
        # Collapse DataProbe widget
        mw = slicer.util.mainWindow()
        if mw:
            w = slicer.util.findChild(mw, "DataProbeCollapsibleWidget")
            if w:
                w.collapsed = True
    
    def addCustomLayouts(self):
        layout2D3D = \
        """
        <layout type="horizontal" split="true">
            <item splitSize="500">
            <view class="vtkMRMLViewNode" singletontag="1">
                <property name="viewlabel" action="default">1</property>
            </view>
            </item>
            <item splitSize="500">
            <view class="vtkMRMLSliceNode" singletontag="Red">
                <property name="orientation" action="default">Axial</property>
                <property name="viewlabel" action="default">R</property>
                <property name="viewcolor" action="default">#F34A33</property>
            </view>
            </item>
        </layout>
        """
         
        layoutManager = slicer.app.layoutManager()
        if not layoutManager.layoutLogic().GetLayoutNode().SetLayoutDescription(self.LAYOUT_2D3D, layout2D3D):
            layoutManager.layoutLogic().GetLayoutNode().AddLayoutDescription(self.LAYOUT_2D3D, layout2D3D)
        
        # Add button to layout selector toolbar for this custom layout
        viewToolBar = slicer.util.mainWindow().findChild("QToolBar", "ViewToolBar")
        layoutMenu = viewToolBar.widgetForAction(viewToolBar.actions()[0]).menu()
        layoutSwitchActionParent = layoutMenu  # use `layoutMenu` to add inside layout list, use `viewToolBar` to add next the standard layout list
        layoutSwitchAction = layoutSwitchActionParent.addAction("3D-2D") # add inside layout list
        layoutSwitchAction.setData(self.LAYOUT_2D3D)
        layoutSwitchAction.setIcon(qt.QIcon(":Icons/Go.png"))
        layoutSwitchAction.setToolTip("3D and slice view")
    
    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        # stop volume reconstruction if running
        if self.logic and self.logic.reconstructing:
            self.logic.stopVolumeReconstruction()
        
        # stop OpenIGTLink connections if running
        if self._parameterNode:
            if self._parameterNode.plusConnectorNode:
                self._parameterNode.plusConnectorNode.Stop()
            if self._parameterNode.predictionConnectorNode:
                self._parameterNode.predictionConnectorNode.Stop()
        
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()
        # Refresh defaults from scene without overriding user-selected nodes.
        self.autoFillKnownSceneNodes()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._onParameterNodeModified)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def onSceneStartImport(self, caller, event) -> None:
        if self.parent.isEntered:
            logging.info("Scene import started: preserving existing hierarchy for recorded-sequence workflows.")
    
    def onSceneEndImport(self, caller, event) -> None:
        if self.parent.isEntered:
            self.logic.setup()
            self.initializeParameterNode()
            self.autoFillKnownSceneNodes()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())
    
    def autoFillKnownSceneNodes(self) -> None:
        """Assign defaults from known recorded-scene names without overriding manual selections."""
        if not hasattr(self, '_parameterNode') or not self._parameterNode or not self.logic:
            return

        parameterNode = self._parameterNode
        nodeCandidates = {
            "inputVolume": [self.logic.IMAGE_IMAGE],
            "predictionVolume": [self.logic.PREDICTION],
            "referenceToRas": [self.logic.REFERENCE_TO_RAS],
            # Recorded scenes often use ProbeToReference while live mode uses ImageToReference.
            "imageToReference": ["ProbeToReference", self.logic.IMAGE_TO_REFERENCE],
            "predictionToReference": [self.logic.PREDICTION_TO_REFERENCE],
            "needleToReference": [self.logic.NEEDLE_TO_REFERENCE],
            "needleTipToneedle": [self.logic.NEEDLE_TIP_TO_NEEDLE],
            "cadProbeToProbe": [self.logic.CAD_PROBE_TO_PROBE],
            "reconstructorNode": [self.logic.RECONSTRUCTOR_NODE],
            "reconstructedVolume": [self.logic.RECONSTRUCTED_VOLUME],
            "kidneyNavMarkups": [self.logic.KIDNEY_NAV_MARKUP],
        }

        for parameterName, candidateNames in nodeCandidates.items():
            if getattr(parameterNode, parameterName):
                continue
            node = self.logic.getFirstNodeByNames(candidateNames)
            if node:
                setattr(parameterNode, parameterName, node)
        
    def setParameterNode(self, inputParameterNode: Optional[KidneyNavParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._onParameterNodeModified)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._onParameterNodeModified)
            self._onParameterNodeModified()

    def _onParameterNodeModified(self, caller=None, event=None) -> None:
        """
        Update GUI based on parameter node changes.
        """
        # Update slice display with input volume
        if self._parameterNode and self._parameterNode.inputVolume:
            slicer.util.setSliceViewerLayers(background=self._parameterNode.inputVolume, fit=True)
            resliceDriverLogic = slicer.modules.volumereslicedriver.logic()
            # Get red slice node
            layoutManager = slicer.app.layoutManager()
            sliceWidget = layoutManager.sliceWidget("Red")
            sliceNode = sliceWidget.mrmlSliceNode()

            # Update slice using reslice driver
            resliceDriverLogic.SetDriverForSlice(self._parameterNode.inputVolume.GetID(), sliceNode)
            # Do not override slice orientation/rotation here.
            # Recorded sequences may already have the desired slice orientation set.

            # Fit slice to background
            sliceWidget.sliceController().fitSliceToBackground()

        # Update volume reconstruction button
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.predictionVolume and self._parameterNode.reconstructorNode:
            if self.logic.reconstructing:
                self.ui.applyButton.text = _("Stop volume reconstruction")
                self.ui.applyButton.toolTip = _("Stop volume reconstruction")
                self.ui.applyButton.checked = True
            else:
                self.ui.applyButton.text = _("Start volume reconstruction")
                self.ui.applyButton.toolTip = _("Start volume reconstruction")
                self.ui.applyButton.checked = False
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select input nodes to enable volume reconstruction")
            self.ui.applyButton.enabled = False
        
        # Update opacity threshold slider and segmentation visualization
        vrLogic = slicer.modules.volumerendering.logic()
        if self._parameterNode and self._parameterNode.reconstructedVolume:
            self.ui.volumeOpacitySlider.enabled = True
            # Update visibility of volumes
            if self.displayedReconstructedVolume and self.displayedReconstructedVolume != self._parameterNode.reconstructedVolume:
                previousDisplayNode = vrLogic.GetFirstVolumeRenderingDisplayNode(self.displayedReconstructedVolume)
                if previousDisplayNode:
                    previousDisplayNode.SetVisibility(False)
            self.displayedReconstructedVolume = self._parameterNode.reconstructedVolume
            currentDisplayNode = vrLogic.GetFirstVolumeRenderingDisplayNode(self.displayedReconstructedVolume)
            if currentDisplayNode:
                currentDisplayNode.SetVisibility(True)

            # Update segmentation visualization when parameter node changes
            if hasattr(self._parameterNode, 'showKidney'):
                self.logic.updateSegmentationVisualization(
                    self._parameterNode.reconstructedVolume,
                    self._parameterNode.showKidney,
                )
        else:
            self.ui.volumeOpacitySlider.enabled = False

        # Set and observe needleToReference transform
        if self.observedKidneyNavMarkups != self._parameterNode.kidneyNavMarkups:
            if self.observedKidneyNavMarkups:
                self.removeObserver(self.observedKidneyNavMarkups, vtkMRMLMarkupsFiducialNode.TransformModifiedEvent, self._onneedleToReferenceModified)
            self.observedKidneyNavMarkups = self._parameterNode.kidneyNavMarkups
            if self.observedKidneyNavMarkups:
                self.addObserver(self.observedKidneyNavMarkups, vtkMRMLMarkupsFiducialNode.TransformModifiedEvent, self._onneedleToReferenceModified)
            self._onneedleToReferenceModified()

        if self._parameterNode and hasattr(self.ui, 'inPlaneDisplayModeComboBox'):
            displayMode = self._parameterNode.inPlaneDisplayMode if self._parameterNode.inPlaneDisplayMode else "Projection"
            comboBox = self.ui.inPlaneDisplayModeComboBox
            modeIndex = comboBox.findText(displayMode)
            if modeIndex < 0:
                modeIndex = 0
            wasBlocked = comboBox.blockSignals(True)
            comboBox.setCurrentIndex(modeIndex)
            comboBox.blockSignals(wasBlocked)
            self.logic.updateInPlaneOverlayDisplay()
        
    def _onneedleToReferenceModified(self, caller=None, event=None) -> None:
        # Distance calculation removed - target points feature has been removed
        pass
    
    def onOpenIGTLinkButton(self, checked: bool) -> None:
        parameterNode = self._parameterNode
        if not parameterNode.plusConnectorNode or not parameterNode.predictionConnectorNode:
            logging.warning("OpenIGTLink connectors are not selected/found. Select them in the scene first.")
            self.ui.startOpenIGTLinkButton.checked = False
            return
        if checked:
            parameterNode.plusConnectorNode.Start()
            parameterNode.predictionConnectorNode.Start()
        else:
            parameterNode.plusConnectorNode.Stop()
            parameterNode.predictionConnectorNode.Stop()
    
    def onReconstructionButton(self) -> None:
        """Run processing when user clicks button."""
        # Start volume reconstruction if not already started. Stop otherwise.
        
        if self.logic.reconstructing:
            self.ui.applyButton.text = _("Start volume reconstruction")
            self.ui.applyButton.toolTip = _("Start volume reconstruction")
            self.ui.applyButton.checked = False
            self.logic.stopVolumeReconstruction()
        else:
            self.ui.applyButton.text = _("Stop volume reconstruction")
            self.ui.applyButton.toolTip = _("Stop volume reconstruction")
            self.ui.applyButton.checked = True
            self.logic.startVolumeReconstruction()
    
    def onVolumeOpacitySlider(self, value: int) -> None:
        """Update volume rendering opacity threshold."""
        if self._parameterNode and self._parameterNode.reconstructedVolume:
            self.logic.setVolumeRenderingProperty(self._parameterNode.reconstructedVolume, window=200, level=(255 - value))
    
    def onSetRoiButton(self) -> None:
        """
        Set volume reconstruction ROI and ReferenceToRas transform based on the current location of the ultrasound image.
        The center of ultrasound will be the center of the ROI. Marked (X) direction of the image will be aligne to Right (R) and Far (Y) to Anterior (A).
        """
        self.logic.resetReferenceToRasBasedOnImage()
        self.logic.resetRoiAndTargetsBasedOnImage()

    def onBlurButton(self) -> None:
        if self._parameterNode and self._parameterNode.reconstructedVolume:
            outputVolume = self.logic.blurVolume(self._parameterNode.reconstructedVolume, self._parameterNode.blurSigma)

            # Set volume property to MR-Default
            vrLogic = slicer.modules.volumerendering.logic()
            outputDisplayNode = vrLogic.CreateDefaultVolumeRenderingNodes(outputVolume)
            outputDisplayNode.GetVolumePropertyNode().Copy(vrLogic.GetPresetByName("MR-Default"))
            outputDisplayNode.SetVisibility(True)

            if self._parameterNode.inputVolume:
                # Change slice view back to Image_Image and reslice
                slicer.util.setSliceViewerLayers(background=self._parameterNode.inputVolume, fit=True)
                resliceDriverLogic = slicer.modules.volumereslicedriver.logic()

                # Get red slice node
                layoutManager = slicer.app.layoutManager()
                sliceWidget = layoutManager.sliceWidget("Red")
                sliceNode = sliceWidget.mrmlSliceNode()

                # Update slice using reslice driver
                resliceDriverLogic.SetDriverForSlice(self._parameterNode.inputVolume.GetID(), sliceNode)
                resliceDriverLogic.SetModeForSlice(resliceDriverLogic.MODE_TRANSVERSE, sliceNode)

                # Fit slice to background
                sliceWidget.sliceController().fitSliceToBackground()

            # Set blurred volume as active volume and hide the original volume
            inputDisplayNode = vrLogic.GetFirstVolumeRenderingDisplayNode(self._parameterNode.reconstructedVolume)
            inputDisplayNode.SetVisibility(False)
            self._parameterNode.reconstructedVolume = outputVolume

    def onInPlaneOverlayToggled(self, checked):
        print(f"In-plane overlay toggled: {checked}")
        if checked:
            print(f"In-plane overlay toggled: {checked}")
            self.logic.showInPlaneDepthLines()
        else:
            print(f"In-plane overlay toggled: {checked}")
            self.logic.hideInPlaneDepthLines()

    def onInPlaneDisplayModeChanged(self, index):
        if not self._parameterNode or not hasattr(self.ui, 'inPlaneDisplayModeComboBox'):
            return

        selectedMode = self.ui.inPlaneDisplayModeComboBox.itemText(index)
        if not selectedMode:
            selectedMode = "Projection"
        self._parameterNode.inPlaneDisplayMode = selectedMode
        self.logic.updateInPlaneOverlayDisplay()

    def onOutOfPlaneOverlayToggled(self, checked):
        print(f"In-plane overlay toggled: {checked}")
    
    def onSegmentationToggled(self, checked):
        """Update volume rendering when segmentation visibility is toggled."""
        if self._parameterNode and self._parameterNode.reconstructedVolume:
            self.logic.updateSegmentationVisualization(
                self._parameterNode.reconstructedVolume,
                self._parameterNode.showKidney,
            )
    
    def onKidneyTypeChanged(self, index):
        """Handle kidney type selection from combobox."""
        if not self._parameterNode:
            return
        
        if index == 0:  # Left kidney
            self._parameterNode.recordingLeft = True
            self._parameterNode.recordingRight = False
        elif index == 1:  # Right kidney
            self._parameterNode.recordingLeft = False
            self._parameterNode.recordingRight = True

    def _getCurrentSequenceName(self) -> str:
        """
        Build the current sequence name based on participant id and task type.
        Format: 3-digit participant id + '_' + task, e.g. '001_manual'.
        """
        participantNum = self.ui.patientNumberSpinBox.value
        participant_id = f"{participantNum:03d}"

        taskText = "manual"
        if hasattr(self.ui, "taskComboBox"):
            try:
                taskText = (self.ui.taskComboBox.currentText or "manual").strip().lower()
            except Exception:
                taskText = "manual"

        # Sanitize for filesystem safety
        safeTask = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in taskText)
        safeTask = safeTask.strip("_") or "manual"

        return f"{participant_id}_{safeTask}"

    def onInitializeRecordingButton(self):
        """Initialize the sequence browser and proxies for the current participant/kidney."""
        print("[Initialize Recording] Button pressed")
        if not self._parameterNode:
            msg = "Parameter node missing; attempting to initialize now"
            print(f"[Initialize Recording] {msg}")
            logging.warning(msg)
            try:
                self.initializeParameterNode()
            except Exception as e:
                err = f"Failed to initialize parameter node: {e}"
                print(f"[Initialize Recording] {err}")
                logging.error(err)
            if not self._parameterNode:
                err = "Parameter node is not available. Try reloading the module."
                print(f"[Initialize Recording] {err}")
                logging.error(err)
                return
        
        if not self._parameterNode.inputVolume:
            msg = "Please select an input volume first"
            print(f"[Initialize Recording] {msg}")
            logging.warning(msg)
            return
        
        sequenceName = self._getCurrentSequenceName()
        print(f"[Initialize Recording] Creating sequence browser: {sequenceName}")
        
        # Create (or reuse) sequence browser and configure proxies
        self.logic.createAndConfigureSequenceBrowser(
            sequenceName,
            self._parameterNode.inputVolume,
            self._parameterNode.predictionVolume
        )
        
        # Connect sequence browser widget to the new node
        self._updateSequenceBrowserWidget()
        
        msg = f"Recording sequence '{sequenceName}' initialized and ready for recording"
        print(f"[Initialize Recording] {msg}")
        logging.info(msg)
    
    def _updateSequenceBrowserWidget(self):
        """Connect the sequence browser widgets to the current sequence browser node."""
        if not self._parameterNode or not self._parameterNode.sequenceBrowserNode:
            return
        
        sequenceBrowserNode = self._parameterNode.sequenceBrowserNode
        
        if hasattr(self.ui, 'sequenceBrowserPlayWidget'):
            self.ui.sequenceBrowserPlayWidget.setMRMLSequenceBrowserNode(sequenceBrowserNode)
        if hasattr(self.ui, 'sequenceBrowserSeekWidget'):
            self.ui.sequenceBrowserSeekWidget.setMRMLSequenceBrowserNode(sequenceBrowserNode)
    
    def onSaveRecordingButton(self):
        """Save the current recording to disk."""
        print("[Save Recording] Button pressed")
        if not self._parameterNode or not self._parameterNode.sequenceBrowserNode:
            msg = "No active recording. Please initialize and record a sequence first."
            print(f"[Save Recording] {msg}")
            logging.warning(msg)
            return
        
        # Get sequence browser and check if it has recorded data
        sequenceBrowserNode = self._parameterNode.sequenceBrowserNode
        synchronizedNodes = vtk.vtkCollection()
        sequenceBrowserNode.GetSynchronizedSequenceNodes(synchronizedNodes, True)
        
        if synchronizedNodes.GetNumberOfItems() == 0:
            msg = "Sequence browser has no synchronized nodes to save."
            print(f"[Save Recording] {msg}")
            logging.warning(msg)
            return

        # Stop recording before saving to avoid partial writes
        try:
            self.logic.stopSequenceRecording()
        except Exception:
            pass

        # Verify we actually have recorded frames (not just configured proxies)
        maxFrames = 0
        for i in range(synchronizedNodes.GetNumberOfItems()):
            seqNode = synchronizedNodes.GetItemAsObject(i)
            if not seqNode:
                continue
            try:
                if hasattr(seqNode, "GetNumberOfDataNodes"):
                    maxFrames = max(maxFrames, int(seqNode.GetNumberOfDataNodes()))
            except Exception:
                continue
        if maxFrames <= 0:
            msg = "No frames recorded yet. Please record a sequence before saving."
            print(f"[Save Recording] {msg}")
            logging.warning(msg)
            return
        
        # Generate base name based on current participant/kidney (e.g., 001_LK)
        sequenceName = self._getCurrentSequenceName()
        
        # Get output folder from UI selector, or use default
        if hasattr(self.ui, 'outputFolderPathLineEdit') and self.ui.outputFolderPathLineEdit.currentPath:
            baseDir = self.ui.outputFolderPathLineEdit.currentPath
        else:
            homeDir = os.path.expanduser("~")
            baseDir = os.path.join(homeDir, "Documents", "KidneyNavRecordings")
        
        participantId = sequenceName.split("_")[0]
        participantDir = os.path.join(baseDir, participantId)
        saveDir = os.path.join(participantDir, sequenceName)
        
        # Create directory if it doesn't exist
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        
        try:
            # Save sequence browser node
            sequenceFilename = os.path.join(saveDir, f"{sequenceName}.mrml")
            slicer.util.saveNode(sequenceBrowserNode, sequenceFilename)

            # Save all synchronized sequence nodes with consistent naming
            recordedProxyNames = []
            for i in range(synchronizedNodes.GetNumberOfItems()):
                sequenceNode = synchronizedNodes.GetItemAsObject(i)
                if sequenceNode:
                    nodeName = sequenceNode.GetName()
                    recordedProxyNames.append(nodeName)
                    nodeFilename = os.path.join(saveDir, f"{sequenceName}_{nodeName}.seq.nrrd")
                    slicer.util.saveNode(sequenceNode, nodeFilename)

            # Write metadata sidecar
            taskValue = None
            if hasattr(self.ui, "taskComboBox"):
                try:
                    taskValue = (self.ui.taskComboBox.currentText or "").strip()
                except Exception:
                    taskValue = None
            if not taskValue:
                parts = sequenceName.split("_", 1)
                taskValue = parts[1] if len(parts) > 1 else "manual"

            meta = {
                "sequenceName": sequenceName,
                "participantId": participantId,
                "task": taskValue,
                "savedAtUtc": datetime.now(timezone.utc).isoformat(),
                "outputDirectory": saveDir,
                "recordedProxies": recordedProxyNames,
                "maxFrames": maxFrames,
            }
            try:
                meta["slicerVersion"] = slicer.app.applicationVersion
            except Exception:
                pass
            metaPath = os.path.join(saveDir, f"{sequenceName}_metadata.json")
            with open(metaPath, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            
            msg = f"Recording '{sequenceName}' saved to {saveDir}"
            print(f"[Save Recording] {msg}")
            logging.info(msg)
        except Exception as e:
            err = f"Failed to save recording: {str(e)}"
            print(f"[Save Recording] {err}")
            logging.error(err)
    
#
# KidneyNavLogic
#


class KidneyNavLogic(ScriptedLoadableModuleLogic):
    """Logic for KidneyNav volume reconstruction and visualization.

    This logic is agnostic to which segmentation checkpoint is used.
    It assumes that an external inference client:
    - Produces a prediction volume node named ``Prediction`` (or wired via the parameter node)
      containing a binary label map where 0=background and 1=target anatomy.
    - Streams that prediction into Slicer over OpenIGTLink.

    KidneyNavLogic is responsible only for:
    - Creating/configuring MRML nodes (transforms, input/prediction volumes, reconstruction, overlays)
    - Driving live volume reconstruction from the prediction volume
    - Configuring binary volume rendering of the reconstructed volume
    """

    # transform names
    REFERENCE_TO_RAS = "ReferenceToRas"
    IMAGE_TO_REFERENCE = "ImageToReference"
    PREDICTION_TO_REFERENCE = "PredToReference"
    NEEDLE_TO_REFERENCE = "NeedleToReference"
    NEEDLE_TIP_TO_NEEDLE = "NeedleTipToNeedle"
    CAD_PROBE_TO_PROBE = "CADProbeToProbe"

    # volume names
    IMAGE_IMAGE = "Image_Image"
    PREDICTION = "Prediction"

    # reconstruction nodes
    RECONSTRUCTOR_NODE = "VolumeReconstruction"
    RECONSTRUCTED_VOLUME = "ReconstructedVolume"
    RECONSTRUCTION_ROI = "ReconstructionROI"

    # OpenIGTLink parameters
    PLUS_CONNECTOR = "PlusConnector"
    PREDICTION_CONNECTOR = "PredictionConnector"
    PLUS_CONNECTOR_PORT = 18944
    PREDICTION_CONNECTOR_PORT = 18945

    # kidney nav parameters
    NEEDLE_MODEL = "NeedleModel"
    NEEDLE_LENGTH = 80  # mm
    KIDNEY_NAV_MARKUP = "KidneyNavMarkup"
    INPLANE_OVERLAY_MODEL = "InPlaneDepthOverlayModel"

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)
        
        self.reconstructing = False
        # Render in-plane guide channels as separate model nodes to allow distinct colors.
        self.inPlaneOverlayModelNodes = {}

    def getParameterNode(self):
        return KidneyNavParameterNode(super().getParameterNode())

    def getFirstNodeByNames(self, names, className=None):
        for nodeName in names:
            node = None
            try:
                node = slicer.util.getNode(nodeName)
            except Exception:
                node = None
            if not node:
                continue
            if className and not node.IsA(className):
                continue
            return node
        return None

    def _getOrCreateNode(self, parameterNode, parameterName, className, primaryName, aliases=None, initializer=None):
        aliases = aliases or []
        node = getattr(parameterNode, parameterName)
        created = False

        if not node:
            node = self.getFirstNodeByNames([primaryName] + aliases, className)
        if not node:
            node = slicer.mrmlScene.AddNewNodeByClass(className, primaryName)
            created = True
            if initializer:
                initializer(node)

        if node:
            setattr(parameterNode, parameterName, node)
        return node, created

    def setup(self):
        # Manual-node workflow: only predictionVolume is auto-created if missing.
        parameterNode = self.getParameterNode()

        parameterNode.referenceToRas = parameterNode.referenceToRas or self.getFirstNodeByNames(
            [self.REFERENCE_TO_RAS], "vtkMRMLLinearTransformNode"
        )
        parameterNode.imageToReference = parameterNode.imageToReference or self.getFirstNodeByNames(
            ["ProbeToReference", self.IMAGE_TO_REFERENCE], "vtkMRMLLinearTransformNode"
        )
        parameterNode.cadProbeToProbe = parameterNode.cadProbeToProbe or self.getFirstNodeByNames(
            [self.CAD_PROBE_TO_PROBE], "vtkMRMLLinearTransformNode"
        )
        parameterNode.inputVolume = parameterNode.inputVolume or self.getFirstNodeByNames(
            [self.IMAGE_IMAGE], "vtkMRMLScalarVolumeNode"
        )
        predictionToReference = parameterNode.predictionToReference or self.getFirstNodeByNames(
            [self.PREDICTION_TO_REFERENCE], "vtkMRMLLinearTransformNode"
        )
        if not predictionToReference:
            predictionToReference = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLLinearTransformNode", self.PREDICTION_TO_REFERENCE
            )
            if parameterNode.referenceToRas and not predictionToReference.GetParentTransformNode():
                predictionToReference.SetAndObserveTransformNodeID(parameterNode.referenceToRas.GetID())
            logging.info("Auto-created missing prediction transform '%s'.", self.PREDICTION_TO_REFERENCE)
        parameterNode.predictionToReference = predictionToReference

        predictionVolume = parameterNode.predictionVolume or self.getFirstNodeByNames(
            [self.PREDICTION], "vtkMRMLScalarVolumeNode"
        )
        predictionVolumeCreated = False
        if not predictionVolume:
            predictionVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", self.PREDICTION)
            predictionVolume.CreateDefaultDisplayNodes()
            predictionArray = np.zeros((1, 512, 512), dtype="uint8")
            slicer.util.updateVolumeFromArray(predictionVolume, predictionArray)
            predictionVolumeCreated = True
            logging.info("Auto-created missing prediction volume '%s'.", self.PREDICTION)
        parameterNode.predictionVolume = predictionVolume
        if predictionVolumeCreated and parameterNode.predictionToReference and not predictionVolume.GetParentTransformNode():
            predictionVolume.SetAndObserveTransformNodeID(parameterNode.predictionToReference.GetID())

        parameterNode.needleToReference = parameterNode.needleToReference or self.getFirstNodeByNames(
            [self.NEEDLE_TO_REFERENCE], "vtkMRMLLinearTransformNode"
        )
        parameterNode.needleTipToneedle = parameterNode.needleTipToneedle or self.getFirstNodeByNames(
            [self.NEEDLE_TIP_TO_NEEDLE], "vtkMRMLLinearTransformNode"
        )
        parameterNode.needleModel = parameterNode.needleModel or self.getFirstNodeByNames(
            [self.NEEDLE_MODEL], "vtkMRMLModelNode"
        )
        parameterNode.reconstructedVolume = parameterNode.reconstructedVolume or self.getFirstNodeByNames(
            [self.RECONSTRUCTED_VOLUME], "vtkMRMLScalarVolumeNode"
        )
        parameterNode.reconstructorNode = parameterNode.reconstructorNode or self.getFirstNodeByNames(
            [self.RECONSTRUCTOR_NODE], "vtkMRMLVolumeReconstructionNode"
        )
        parameterNode.kidneyNavMarkups = parameterNode.kidneyNavMarkups or self.getFirstNodeByNames(
            [self.KIDNEY_NAV_MARKUP], "vtkMRMLMarkupsFiducialNode"
        )

        if not parameterNode.reconstructedVolume:
            parameterNode.reconstructedVolume = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLScalarVolumeNode", self.RECONSTRUCTED_VOLUME
            )
            parameterNode.reconstructedVolume.CreateDefaultDisplayNodes()

        if not parameterNode.reconstructorNode:
            parameterNode.reconstructorNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLVolumeReconstructionNode", self.RECONSTRUCTOR_NODE
            )
            parameterNode.reconstructorNode.SetLiveVolumeReconstruction(True)
            parameterNode.reconstructorNode.SetInterpolationMode(1)  # linear
            logging.info("Auto-created missing volume reconstruction node '%s'.", self.RECONSTRUCTOR_NODE)

        if parameterNode.needleModel and parameterNode.needleModel.GetDisplayNode():
            parameterNode.needleModel.GetDisplayNode().Visibility2DOn()
        if parameterNode.needleModel and parameterNode.needleTipToneedle and not parameterNode.needleModel.GetParentTransformNode():
            parameterNode.needleModel.SetAndObserveTransformNodeID(parameterNode.needleTipToneedle.GetID())

        if parameterNode.reconstructedVolume:
            volRenLogic = slicer.modules.volumerendering.logic()
            reconstructedDisplay = volRenLogic.GetFirstVolumeRenderingDisplayNode(parameterNode.reconstructedVolume)
            if not reconstructedDisplay:
                reconstructedDisplay = volRenLogic.CreateDefaultVolumeRenderingNodes(parameterNode.reconstructedVolume)
            reconstructedDisplay.SetVisibility(True)
            reconstructedDisplay.GetVolumePropertyNode().Copy(volRenLogic.GetPresetByName("MR-Default"))

        if parameterNode.reconstructorNode:
            parameterNode.reconstructorNode.SetLiveVolumeReconstruction(True)
            parameterNode.reconstructorNode.SetInterpolationMode(1)  # linear
            if parameterNode.predictionVolume and not parameterNode.reconstructorNode.GetInputVolumeNode():
                parameterNode.reconstructorNode.SetAndObserveInputVolumeNode(parameterNode.predictionVolume)
            if parameterNode.reconstructedVolume and not parameterNode.reconstructorNode.GetOutputVolumeNode():
                parameterNode.reconstructorNode.SetAndObserveOutputVolumeNode(parameterNode.reconstructedVolume)
            if not parameterNode.reconstructorNode.GetInputROINode():
                roiNode = self.getFirstNodeByNames([self.RECONSTRUCTION_ROI], "vtkMRMLMarkupsROINode")
                if not roiNode:
                    roiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", self.RECONSTRUCTION_ROI)
                    roiNode.SetSize((250, 250, 350))
                    roiNode.SetDisplayVisibility(False)
                parameterNode.reconstructorNode.SetAndObserveInputROINode(roiNode)

        if parameterNode.kidneyNavMarkups:
            parameterNode.kidneyNavMarkups.SetMaximumNumberOfControlPoints(1)
            parameterNode.kidneyNavMarkups.CreateDefaultDisplayNodes()
            parameterNode.kidneyNavMarkups.SetDisplayVisibility(False)

        missingManualNodes = []
        for parameterName in [
            "referenceToRas",
            "imageToReference",
            "cadProbeToProbe",
            "inputVolume",
            "predictionToReference",
            "needleToReference",
            "needleTipToneedle",
            "needleModel",
            "reconstructedVolume",
            "reconstructorNode",
            "kidneyNavMarkups",
        ]:
            if not getattr(parameterNode, parameterName):
                missingManualNodes.append(parameterName)
        if missingManualNodes:
            logging.warning(
                "Manual scene nodes not found (not auto-created): %s",
                ", ".join(missingManualNodes),
            )

        self.setupOpenIgtLink()
    
    def setupOpenIgtLink(self):
        parameterNode = self.getParameterNode()
        parameterNode.plusConnectorNode = parameterNode.plusConnectorNode or self.getFirstNodeByNames(
            [self.PLUS_CONNECTOR], "vtkMRMLIGTLConnectorNode"
        )
        parameterNode.predictionConnectorNode = parameterNode.predictionConnectorNode or self.getFirstNodeByNames(
            [self.PREDICTION_CONNECTOR], "vtkMRMLIGTLConnectorNode"
        )

        if not parameterNode.plusConnectorNode:
            logging.warning("PLUS connector '%s' not found (not auto-created).", self.PLUS_CONNECTOR)
        if not parameterNode.predictionConnectorNode:
            logging.warning("Prediction connector '%s' not found (not auto-created).", self.PREDICTION_CONNECTOR)
    
    def startVolumeReconstruction(self):
        """
        Start live volume reconstruction.
        """
        parameterNode = self.getParameterNode()
        if not parameterNode.reconstructorNode:
            logging.warning("Volume reconstruction node is not selected/found. Cannot start reconstruction.")
            return
        self.reconstructing = True
        reconstructionLogic = slicer.modules.volumereconstruction.logic()
        reconstructionLogic.StartLiveVolumeReconstruction(parameterNode.reconstructorNode)
        outputVolume = parameterNode.reconstructorNode.GetOutputVolumeNode()
        # Use binary segmentation visualization (single foreground label)
        self.updateSegmentationVisualization(
            outputVolume,
            parameterNode.showKidney,
        )
        parameterNode.reconstructedVolume = outputVolume
    
    def stopVolumeReconstruction(self):
        """
        Stop live volume reconstruction.
        """
        parameterNode = self.getParameterNode()
        self.reconstructing = False
        reconstructionLogic = slicer.modules.volumereconstruction.logic()
        reconstructionLogic.StopLiveVolumeReconstruction(parameterNode.reconstructorNode)
    
    def setVolumeRenderingProperty(self, volumeNode, window=255, level=127):
        volumeRenderingLogic = slicer.modules.volumerendering.logic()
        volumeRenderingDisplayNode = volumeRenderingLogic.GetFirstVolumeRenderingDisplayNode(volumeNode)
        if not volumeRenderingDisplayNode:
            volumeRenderingDisplayNode = volumeRenderingLogic.CreateDefaultVolumeRenderingNodes(volumeNode)
            
        upper = min(255 + window, level + window/2)
        lower = max(0 - window, level - window/2)

        if upper <= lower:
            upper = lower + 1  # Make sure the displayed intensity range is valid.

        p0 = lower
        p1 = lower + (upper - lower)*0.15
        p2 = lower + (upper - lower)*0.4
        p3 = upper

        opacityTransferFunction = vtk.vtkPiecewiseFunction()
        opacityTransferFunction.AddPoint(p0, 0.0)
        opacityTransferFunction.AddPoint(p1, 0.2)
        opacityTransferFunction.AddPoint(p2, 0.6)
        opacityTransferFunction.AddPoint(p3, 1)

        colorTransferFunction = vtk.vtkColorTransferFunction()
        colorTransferFunction.AddRGBPoint(p0, 0.20, 0.10, 0.00)
        colorTransferFunction.AddRGBPoint(p1, 0.65, 0.45, 0.15)
        colorTransferFunction.AddRGBPoint(p2, 0.85, 0.75, 0.55)
        colorTransferFunction.AddRGBPoint(p3, 1.00, 1.00, 0.80)

        volumeProperty = volumeRenderingDisplayNode.GetVolumePropertyNode().GetVolumeProperty()
        volumeProperty.SetColor(colorTransferFunction)
        volumeProperty.SetScalarOpacity(opacityTransferFunction)
        volumeProperty.ShadeOn()
        volumeProperty.SetInterpolationTypeToLinear()
    
    def updateSegmentationVisualization(self, volumeNode, visible=True):
        """
        Update volume rendering to show a single binary segmentation class.

        Assumes:
        - 0: Background (always transparent)
        - 1: Target anatomy (kidney or renal pelvis)
        """
        if not volumeNode:
            return

        volumeRenderingLogic = slicer.modules.volumerendering.logic()
        volumeRenderingDisplayNode = volumeRenderingLogic.GetFirstVolumeRenderingDisplayNode(volumeNode)
        if not volumeRenderingDisplayNode:
            volumeRenderingDisplayNode = volumeRenderingLogic.CreateDefaultVolumeRenderingNodes(volumeNode)

        if not visible:
            volumeRenderingDisplayNode.SetVisibility(False)
            return

        # Create color transfer function for binary segmentation
        colorTransferFunction = vtk.vtkColorTransferFunction()
        opacityTransferFunction = vtk.vtkPiecewiseFunction()

        # Background: fully transparent
        colorTransferFunction.AddRGBPoint(0, 0.0, 0.0, 0.0)
        opacityTransferFunction.AddPoint(0, 0.0)

        # Foreground label 1: solid color
        colorTransferFunction.AddRGBPoint(1, 0.8, 0.2, 0.2)  # reddish
        opacityTransferFunction.AddPoint(1, 0.9)

        volumeProperty = volumeRenderingDisplayNode.GetVolumePropertyNode().GetVolumeProperty()
        volumeProperty.SetColor(colorTransferFunction)
        volumeProperty.SetScalarOpacity(opacityTransferFunction)
        volumeProperty.ShadeOn()
        volumeProperty.SetInterpolationTypeToLinear()

        volumeRenderingDisplayNode.SetVisibility(True)
    
    def resetReferenceToRasBasedOnImage(self):
        """
        Get the current position of Image in RAS. Make sure ReferenceToRas transform is aligned with the image.
        Image should be aligned so X is Right, and Y is Anterior.
        """
        parameterNode = self.getParameterNode()
        
        inputVolume = parameterNode.inputVolume
        if not inputVolume:
            logging.error("Input volume is not set")
            return
        
        referenceToRas = parameterNode.referenceToRas
        if not referenceToRas:
            logging.error("ReferenceToRas transform is not set")
            return
        
        # Temporarily set ReferenceToRas matrix to identity
        referenceToRasMatrix = vtk.vtkMatrix4x4()
        referenceToRas.GetMatrixTransformToWorld(referenceToRasMatrix)
        referenceToRasMatrix.Identity()
        referenceToRas.SetMatrixTransformToParent(referenceToRasMatrix)
        
        # Get the current position of Image in RAS
        imageToReferenceTransform = slicer.mrmlScene.GetNodeByID(inputVolume.GetTransformNodeID())
        if imageToReferenceTransform is None:
            logging.error("Image transform is not set")
            return
        
        imageToReferenceMatrix = vtk.vtkMatrix4x4()
        imageToReferenceTransform.GetMatrixTransformToWorld(imageToReferenceMatrix)
        imageToReferenceMatrix.Invert()
        # Keep only rotation part from imageToReferenceMatrix to align ReferenceToRas with the image
        referenceToImageTransform = vtk.vtkTransform()
        referenceToImageTransform.SetMatrix(imageToReferenceMatrix)
        referenceToRasTransform = vtk.vtkTransform()
        wxyz = referenceToImageTransform.GetOrientationWXYZ()
        referenceToRasTransform.RotateWXYZ(wxyz[0], wxyz[1], wxyz[2], wxyz[3])
        referenceToRas.SetMatrixTransformToParent(referenceToRasTransform.GetMatrix())
    
    def resetRoiAndTargetsBasedOnImage(self):
        """
        Get the current position of Image in RAS. Make sure volume reconstruction has a ROI node and it is centered in the image.
        """
        parameterNode = self.getParameterNode()
        if not parameterNode.reconstructorNode:
            logging.error("Reconstructor node is not set")
            return
        
        # Get the current position of Image in RAS
        imageNode = parameterNode.inputVolume
        if not imageNode:
            logging.warning("Cannot set ROI because input volume is not set")
            return
        
        # Get the center of the image
        imageBounds_Ras = np.zeros(6)
        imageNode.GetRASBounds(imageBounds_Ras)
        imageCenter_Ras = np.zeros(3)
        for i in range(3):
            imageCenter_Ras[i] = (imageBounds_Ras[i*2] + imageBounds_Ras[i*2+1]) / 2
        
        # Set the center of the ROI to the center of the image
        roiNode = parameterNode.reconstructorNode.GetInputROINode()
        if not roiNode:
            logging.warning("No ROI node found in volume reconstruction node")
            return
        roiNode.SetCenterWorld(imageCenter_Ras)

    def blurVolume(self, inputVolume, sigma):
        parameterNode = self.getParameterNode()

        # Set CLI parameters
        inputVolumeName = inputVolume.GetName()
        outputVolumeName = f"{inputVolumeName}_blurred_{sigma:.2f}"
        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", outputVolumeName)
        parameters = {
            "inputVolume": inputVolume, 
            "outputVolume": outputVolume,
            "sigma": sigma
        }
        
        # Run CLI module
        gaussianBlur = slicer.modules.gaussianblurimagefilter
        cliNode = slicer.cli.runSync(gaussianBlur, None, parameters)

        # Process results
        if cliNode.GetStatus() & cliNode.ErrorsMask:
            errorText = cliNode.GetErrorText()
            logging.error(f"Error in GaussianBlurImageFilter: {errorText}")
            slicer.mrmlScene.RemoveNode(cliNode)
        else:
            slicer.mrmlScene.RemoveNode(cliNode)
            return outputVolume
    
    def createAndConfigureSequenceBrowser(self, sequenceName, inputVolume, predictionVolume):
        """
        Create and configure a sequence browser for recording ultrasound sequences.
        Add proxy nodes for all tracked volumes and models.
        
        :param sequenceName: Name for the sequence (e.g., "P01_LK")
        :param inputVolume: Input ultrasound volume node
        :param predictionVolume: Prediction volume node
        """
        parameterNode = self.getParameterNode()

        browserName = f"SequenceBrowser_{sequenceName}"
        sequenceBrowserNode = None
        try:
            sequenceBrowserNode = slicer.util.getNode(browserName)
            if sequenceBrowserNode and not sequenceBrowserNode.IsA("vtkMRMLSequenceBrowserNode"):
                sequenceBrowserNode = None
        except Exception:
            sequenceBrowserNode = None

        # Create a new sequence browser node if it doesn't exist
        if not sequenceBrowserNode:
            sequenceBrowserNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLSequenceBrowserNode", browserName
            )
        
        # Get sequences logic
        sequencesLogic = slicer.modules.sequences.logic()
        
        # Track success of adding nodes
        successCount = 0
        failedLabels = []
        addedNodeIds = set()

        def add_proxy(node, label):
            nonlocal successCount
            if node:
                nodeId = node.GetID() if hasattr(node, "GetID") else None
                if nodeId and nodeId in addedNodeIds:
                    return
                try:
                    seqNode = sequencesLogic.AddSynchronizedNode(None, node, sequenceBrowserNode)
                    sequenceBrowserNode.SetRecording(seqNode, True)
                    successCount += 1
                    if nodeId:
                        addedNodeIds.add(nodeId)
                    logging.info(f"Added '{label}' proxy node to sequence browser")
                except Exception as e:
                    failedLabels.append(label)
                    logging.error(f"Failed to add '{label}' to sequence browser: {str(e)}")
            else:
                failedLabels.append(label)
                logging.warning(f"Node '{label}' not found or not set")

        # Prefer parameter node references (robust to naming)
        add_proxy(parameterNode.referenceToRas, 'ReferenceToRas')
        add_proxy(parameterNode.imageToReference, 'ImageToReference')
        # Some recorded/live scenes have ProbeToReference as a separate transform. Record it explicitly if present.
        probeToReference = self.getFirstNodeByNames(["ProbeToReference"], "vtkMRMLLinearTransformNode")
        if probeToReference and parameterNode.imageToReference and probeToReference.GetID() == parameterNode.imageToReference.GetID():
            probeToReference = None
        add_proxy(probeToReference, "ProbeToReference")

        # Optional manual correction and CAD correction transforms (appear in study transform tree)
        manualCorrection = self.getFirstNodeByNames(["Manual_Correction", "ManualCorrection"], "vtkMRMLLinearTransformNode")
        add_proxy(manualCorrection, "Manual_Correction")
        correctionCad2Probe = self.getFirstNodeByNames(["Correction_CAD2Probe", "CorrectionCAD2Probe", "CAD2Probe"], "vtkMRMLLinearTransformNode")
        add_proxy(correctionCad2Probe, "Correction_CAD2Probe")
        add_proxy(parameterNode.cadProbeToProbe, "CADProbeToProbe")

        add_proxy(parameterNode.predictionToReference, 'PredToReference')
        add_proxy(parameterNode.inputVolume or inputVolume, 'InputVolume')
        # Record the canonical ultrasound stream node name too (even if a different input is selected)
        imageImage = self.getFirstNodeByNames([self.IMAGE_IMAGE], "vtkMRMLScalarVolumeNode")
        add_proxy(imageImage, self.IMAGE_IMAGE)
        add_proxy(parameterNode.predictionVolume or predictionVolume, 'PredictionVolume')
        add_proxy(parameterNode.needleToReference, 'NeedleToReference')
        add_proxy(parameterNode.needleTipToneedle, 'NeedleTipToNeedle')
        add_proxy(parameterNode.needleModel, 'NeedleModel')

        # No additional scene-named nodes needed; input volume is already added via parameter
        
        # Store sequence browser in parameter node
        parameterNode.sequenceBrowserNode = sequenceBrowserNode

        # Log final status
        if failedLabels:
            msg = f"Sequence browser '{sequenceName}' created with {successCount} nodes. Failed or missing: {', '.join(failedLabels)}."
            print(f"[Sequence Browser] {msg}")
            logging.warning(msg)
        else:
            msg = f"Sequence browser '{sequenceName}' created successfully with all {successCount} proxy nodes!"
            print(f"[Sequence Browser] {msg}")
            logging.info(msg)
        
    def getSequenceBrowserNode(self):
        """Get the current sequence browser node."""
        parameterNode = self.getParameterNode()
        return parameterNode.sequenceBrowserNode
        
    def startSequenceRecording(self):
        """Start recording to the current sequence browser."""
        parameterNode = self.getParameterNode()
        sequenceBrowserNode = parameterNode.sequenceBrowserNode
        
        if not sequenceBrowserNode:
            raise RuntimeError("No active sequence browser node found")
        
        # Get all synchronized sequence nodes
        synchronizedNodes = vtk.vtkCollection()
        sequenceBrowserNode.GetSynchronizedSequenceNodes(synchronizedNodes, True)
        
        if synchronizedNodes.GetNumberOfItems() == 0:
            raise RuntimeError("No synchronized nodes found in sequence browser")
        
        # Set all synchronized sequence nodes to recording mode
        for i in range(synchronizedNodes.GetNumberOfItems()):
            sequenceNode = synchronizedNodes.GetItemAsObject(i)
            sequenceBrowserNode.SetRecording(sequenceNode, True)
        
        logging.info(f"Started sequence recording with {synchronizedNodes.GetNumberOfItems()} proxy nodes")
        
    def stopSequenceRecording(self):
        """Stop recording to the current sequence browser."""
        parameterNode = self.getParameterNode()
        if parameterNode.sequenceBrowserNode:
            # Set all synchronized sequence nodes to non-recording mode
            synchronizedNodes = vtk.vtkCollection()
            parameterNode.sequenceBrowserNode.GetSynchronizedSequenceNodes(synchronizedNodes, True)
            for i in range(synchronizedNodes.GetNumberOfItems()):
                sequenceNode = synchronizedNodes.GetItemAsObject(i)
                parameterNode.sequenceBrowserNode.SetRecording(sequenceNode, False)
            logging.info("Stopped sequence recording")
        
    def showInPlaneDepthLines(self):
        cadProbeToProbe = self._getRequiredCadProbeToProbeTransform()
        if not cadProbeToProbe:
            for node in self.inPlaneOverlayModelNodes.values():
                try:
                    node.SetDisplayVisibility(False)
                except Exception:
                    pass
            return

        # Hide any legacy single-model overlay node (older versions rendered both channels in one model)
        try:
            legacyNode = slicer.util.getNode(self.INPLANE_OVERLAY_MODEL)
            if legacyNode and legacyNode.IsA("vtkMRMLModelNode"):
                legacyNode.SetDisplayVisibility(False)
        except Exception:
            pass

        self._createInPlaneDepthLinesModels()
        for node in self.inPlaneOverlayModelNodes.values():
            node.SetAndObserveTransformNodeID(cadProbeToProbe.GetID())
            node.SetDisplayVisibility(True)

    def hideInPlaneDepthLines(self):
        # Hide any legacy node too
        try:
            legacyNode = slicer.util.getNode(self.INPLANE_OVERLAY_MODEL)
            if legacyNode and legacyNode.IsA("vtkMRMLModelNode"):
                legacyNode.SetDisplayVisibility(False)
        except Exception:
            pass
        for node in self.inPlaneOverlayModelNodes.values():
            try:
                node.SetDisplayVisibility(False)
            except Exception:
                pass

    def updateInPlaneOverlayDisplay(self):
        if not self.inPlaneOverlayModelNodes:
            return
        for name, node in self.inPlaneOverlayModelNodes.items():
            displayNode = node.GetDisplayNode() if node else None
            if displayNode:
                self._applyInPlaneOverlayDisplayStyle(displayNode, channelName=name)

    def _getRequiredCadProbeToProbeTransform(self):
        parameterNode = self.getParameterNode()
        cadProbeToProbe = parameterNode.cadProbeToProbe if parameterNode and parameterNode.cadProbeToProbe else None
        if not cadProbeToProbe:
            cadProbeToProbe = self.getFirstNodeByNames([self.CAD_PROBE_TO_PROBE], "vtkMRMLLinearTransformNode")
        if not cadProbeToProbe:
            logging.warning(
                "In-plane overlay is disabled because required CAD probe transform is not selected/found ('%s').",
                self.CAD_PROBE_TO_PROBE,
            )
        return cadProbeToProbe

    def _appendDottedLineSegments(self, points, lines, pointId, p1, p2,
                                  spacing=6.0, segmentLength=4.0,
                                  extendStart=10.0, extendEnd=80.0):
        p1 = np.array(p1, dtype=float)
        p2 = np.array(p2, dtype=float)
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length <= 1e-6:
            return pointId

        direction = direction / length
        p1Ext = p1 - direction * extendStart
        p2Ext = p2 + direction * extendEnd
        extendedLength = np.linalg.norm(p2Ext - p1Ext)

        t = 0.0
        while t < extendedLength:
            start = p1Ext + direction * t
            end = p1Ext + direction * min(t + segmentLength, extendedLength)

            points.InsertNextPoint(start.tolist())
            points.InsertNextPoint(end.tolist())

            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, pointId)
            line.GetPointIds().SetId(1, pointId + 1)
            lines.InsertNextCell(line)

            pointId += 2
            t += spacing
        return pointId

    def _applyInPlaneOverlayDisplayStyle(self, displayNode, channelName=None):
        parameterNode = self.getParameterNode()
        displayMode = parameterNode.inPlaneDisplayMode if parameterNode.inPlaneDisplayMode else "Projection"

        displayNode.SetVisibility2D(True)
        displayNode.SetLineWidth(4)
        displayNode.SetOpacity(0.9)
        displayNode.SetSliceIntersectionThickness(3)
        # Two distinct channel guide colors (study requirement).
        # Channel1: warm yellow, Channel2: cyan.
        if channelName and channelName.lower().endswith("channel2"):
            displayNode.SetColor(0.10, 0.90, 0.95)
        else:
            displayNode.SetColor(0.98, 0.86, 0.20)

        if displayMode == "Intersection":
            displayNode.SetSliceDisplayModeToIntersection()
        else:
            displayNode.SetSliceDisplayModeToProjection()

    def _createInPlaneDepthLinesModels(self):
        """Create dotted in-plane channel lines in CADProbeToProbe coordinates (one model per channel)."""
        channelDefinitions = [
            ("Channel1", [40.375, -15.363, -0.870], [26.759, -38.103, -0.979]),
            ("Channel2", [45.193, -13.878, -0.939], [38.849, -34.472, -1.044]),
        ]

        for name, p1, p2 in channelDefinitions:
            modelName = f"{self.INPLANE_OVERLAY_MODEL}_{name}"
            modelNode = self.inPlaneOverlayModelNodes.get(name)
            if not modelNode:
                modelNode = None
                try:
                    modelNode = slicer.util.getNode(modelName)
                    if modelNode and not modelNode.IsA("vtkMRMLModelNode"):
                        modelNode = None
                except Exception:
                    modelNode = None
                if not modelNode:
                    modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", modelName)
                self.inPlaneOverlayModelNodes[name] = modelNode

            points = vtk.vtkPoints()
            lines = vtk.vtkCellArray()
            self._appendDottedLineSegments(points, lines, 0, p1, p2)

            polyData = vtk.vtkPolyData()
            polyData.SetPoints(points)
            polyData.SetLines(lines)
            modelNode.SetAndObservePolyData(polyData)

            displayNode = modelNode.GetDisplayNode()
            if not displayNode:
                displayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
                modelNode.SetAndObserveDisplayNodeID(displayNode.GetID())
            self._applyInPlaneOverlayDisplayStyle(displayNode, channelName=name)

        totalSegments = 0
        for node in self.inPlaneOverlayModelNodes.values():
            try:
                pd = node.GetPolyData()
                if pd and pd.GetLines():
                    totalSegments += pd.GetLines().GetNumberOfCells()
            except Exception:
                pass
        logging.info(
            "InPlaneDepthOverlay: rendered %d channels (%d dotted segments) in CADProbeToProbe",
            len(channelDefinitions),
            totalSegments,
        )



#
# KidneyNavTest
#


class KidneyNavTest(ScriptedLoadableModuleTest):
    """
    Basic smoke test for the KidneyNav module.

    This does not exercise the full reconstruction pipeline (which depends on
    external OpenIGTLink streams), but ensures that the logic can be instantiated
    and that core MRML nodes are created without errors.
    """

    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_KidneyNav_LogicSetup()

    def test_KidneyNav_LogicSetup(self):
        logic = KidneyNavLogic()
        logic.setup()
        parameterNode = logic.getParameterNode()
        self.assertIsNotNone(parameterNode.inputVolume)
        self.assertIsNotNone(parameterNode.predictionVolume)
        self.assertIsNotNone(parameterNode.reconstructorNode)
