import logging
import numpy as np
import os
import math
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
    vtkMRMLIGTLConnectorNode
)


#
# NeedleGuide
#


class NeedleGuide(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("NeedleGuide")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Ultrasound")]
        self.parent.dependencies = ["VolumeResliceDriver"]  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Tamas Ungi (Queen's University)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#NeedleGuide">module documentation</a>.
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

    # NeedleGuide1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="NeedleGuide",
        sampleName="NeedleGuide1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "NeedleGuide1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="NeedleGuide1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="NeedleGuide1",
    )

    # NeedleGuide2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="NeedleGuide",
        sampleName="NeedleGuide2",
        thumbnailFileName=os.path.join(iconsPath, "NeedleGuide2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="NeedleGuide2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="NeedleGuide2",
    )


#
# NeedleGuideParameterNode
#


@parameterNodeWrapper
class NeedleGuideParameterNode:
    """
    The parameters needed by module.
    """
    inputVolume: vtkMRMLScalarVolumeNode
    referenceToRas: vtkMRMLLinearTransformNode
    imageToReference: vtkMRMLLinearTransformNode
    stylusToReference: vtkMRMLLinearTransformNode
    stylusTipToStylus: vtkMRMLLinearTransformNode
    needleModel: vtkMRMLModelNode
    predictionToReference: vtkMRMLLinearTransformNode
    predictionVolume: vtkMRMLScalarVolumeNode
    reconstructorNode: vtkMRMLVolumeReconstructionNode
    needleGuideMarkups: vtkMRMLMarkupsFiducialNode
    plusConnectorNode: vtkMRMLIGTLConnectorNode
    predictionConnectorNode: vtkMRMLIGTLConnectorNode
    blurSigma: Annotated[float, WithinRange(0, 5)] = 0.5
    reconstructedVolume: vtkMRMLScalarVolumeNode
    opacityThreshold: Annotated[int, WithinRange(-100, 200)] = 60
    invertThreshold: bool = False
    showKidney: bool = True
    showCalyx: bool = True

#
# NeedleGuideWidget
#

class NeedleGuideWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
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
        self.observedNeedleGuideMarkups = None

        # for debugging
        slicer.mymod = self

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/NeedleGuide.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = NeedleGuideLogic()
        self.logic.setup()

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
        
        # Connect segmentation class visualization checkboxes
        self.ui.showKidneyCheckBox.connect('toggled(bool)', self.onSegmentationClassToggled)
        self.ui.showCalyxCheckBox.connect('toggled(bool)', self.onSegmentationClassToggled)
       
        # Add custom layout
        self.addCustomLayouts()
        slicer.app.layoutManager().setLayout(self.LAYOUT_2D3D)
        slicer.app.layoutManager().sliceWidget("Red").sliceController().setSliceVisible(True)
        for viewNode in slicer.util.getNodesByClass("vtkMRMLAbstractViewNode"):
            viewNode.SetOrientationMarkerType(slicer.vtkMRMLAbstractViewNode.OrientationMarkerTypeHuman)
        
        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()
        
        # Auto-assign hardcoded nodes (bypasses UI dropdowns)
        self.autoAssignHardcodedNodes()
        
        # Hide node selector dropdowns from UI (they work in background)
        self.hideNodeSelectors()
        
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
        if self.logic.reconstructing:
            self.logic.stopVolumeReconstruction()
        
        # stop OpenIGTLink connections if running
        if self._parameterNode:
            self._parameterNode.plusConnectorNode.Stop()
            self._parameterNode.predictionConnectorNode.Stop()
        
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()
        # Re-assign hardcoded nodes to ensure they're always set correctly
        self.autoAssignHardcodedNodes()

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
            slicer.mrmlScene.Clear(0)
    
    def onSceneEndImport(self, caller, event) -> None:
        if self.parent.isEntered:
            self.logic.setup()
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())
    
    def autoAssignHardcodedNodes(self) -> None:
        """
        Automatically assign nodes by their hardcoded names, bypassing UI dropdowns.
        This ensures nodes are always assigned without user selection.
        """
        if not hasattr(self, '_parameterNode') or not self._parameterNode:
            return
        
        parameterNode = self._parameterNode
        if not hasattr(self, 'logic') or not self.logic:
            return
        
        logic = self.logic
        
        # Assign nodes by their hardcoded names
        try:
            inputVolume = slicer.util.getNode(logic.IMAGE_IMAGE)
            if inputVolume:
                parameterNode.inputVolume = inputVolume
        except Exception as e:
            logging.warning(f"Could not assign inputVolume: {e}")
        
        try:
            predictionVolume = slicer.util.getNode(logic.PREDICTION)
            if predictionVolume:
                parameterNode.predictionVolume = predictionVolume
        except Exception as e:
            logging.warning(f"Could not assign predictionVolume: {e}")
        
        try:
            referenceToRas = slicer.util.getNode(logic.REFERENCE_TO_RAS)
            if referenceToRas:
                parameterNode.referenceToRas = referenceToRas
        except Exception as e:
            logging.warning(f"Could not assign referenceToRas: {e}")
        
        try:
            stylusToReference = slicer.util.getNode(logic.STYLUS_TO_REFERENCE)
            if stylusToReference:
                parameterNode.stylusToReference = stylusToReference
        except Exception as e:
            logging.warning(f"Could not assign stylusToReference: {e}")
        
        try:
            reconstructorNode = slicer.util.getNode(logic.RECONSTRUCTOR_NODE)
            if reconstructorNode:
                parameterNode.reconstructorNode = reconstructorNode
        except Exception as e:
            logging.warning(f"Could not assign reconstructorNode: {e}")
        
        try:
            needleGuideMarkups = slicer.util.getNode(logic.NEEDLE_GUIDE_MARKUP)
            if needleGuideMarkups:
                parameterNode.needleGuideMarkups = needleGuideMarkups
        except Exception as e:
            logging.warning(f"Could not assign needleGuideMarkups: {e}")
    
    def hideNodeSelectors(self) -> None:
        """
        Hide node selector dropdowns and their labels from the UI.
        Nodes are now hardcoded and auto-assigned in the background.
        """
        if not hasattr(self, 'ui') or not self.ui:
            return
        
        # Map of selector widget names to their label widget names
        # These will be completely hidden from the UI
        selectorsToHide = {
            'inputSelector': 'label',                    # Ultrasound image
            'referenceToRasComboBox': 'label_4',        # ReferenceToRas transform
            'stylusToReferenceComboBox': 'label_12',     # StylusToReference transform
            'predictionVolumeComboBox': 'label_6',       # Prediction image
            'reconstructorNodeComboBox': 'label_7',      # Reconstructor node
            'needleGuideMarkupsComboxBox': 'label_11',   # Needle guide markup
        }
        
        # Hide both the selector and its label
        for selectorName, labelName in selectorsToHide.items():
            try:
                # Hide the selector widget
                if hasattr(self.ui, selectorName):
                    selector = getattr(self.ui, selectorName)
                    if selector:
                        selector.setVisible(False)
                        selector.hide()
                        selector.setEnabled(False)
                
                # Hide the label widget
                if hasattr(self.ui, labelName):
                    label = getattr(self.ui, labelName)
                    if label:
                        label.setVisible(False)
                        label.hide()
            except Exception as e:
                logging.warning(f"Could not hide {selectorName}: {e}")
        
    def setParameterNode(self, inputParameterNode: Optional[NeedleGuideParameterNode]) -> None:
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
            resliceDriverLogic.SetModeForSlice(resliceDriverLogic.MODE_TRANSVERSE, sliceNode)
            resliceDriverLogic.SetRotationForSlice(180, sliceNode)

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
            
            # Update segmentation class visualization when parameter node changes
            if hasattr(self._parameterNode, 'showKidney'):
                self.logic.updateSegmentationClassVisualization(
                    self._parameterNode.reconstructedVolume,
                    self._parameterNode.showKidney,
                    self._parameterNode.showCalyx
                )
        else:
            self.ui.volumeOpacitySlider.enabled = False

        # Set and observe StylusToReference transform
        if self.observedNeedleGuideMarkups != self._parameterNode.needleGuideMarkups:
            if self.observedNeedleGuideMarkups:
                self.removeObserver(self.observedNeedleGuideMarkups, vtkMRMLMarkupsFiducialNode.TransformModifiedEvent, self._onStylusToReferenceModified)
            self.observedNeedleGuideMarkups = self._parameterNode.needleGuideMarkups
            if self.observedNeedleGuideMarkups:
                self.addObserver(self.observedNeedleGuideMarkups, vtkMRMLMarkupsFiducialNode.TransformModifiedEvent, self._onStylusToReferenceModified)
            self._onStylusToReferenceModified()
        
    def _onStylusToReferenceModified(self, caller=None, event=None) -> None:
        # Distance calculation removed - target points feature has been removed
        pass
    
    def onOpenIGTLinkButton(self, checked: bool) -> None:
        parameterNode = self._parameterNode
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

    def onOutOfPlaneOverlayToggled(self, checked):
        print(f"In-plane overlay toggled: {checked}")
    
    def onSegmentationClassToggled(self, checked):
        """Update volume rendering when segmentation class visibility is toggled."""
        if self._parameterNode and self._parameterNode.reconstructedVolume:
            self.logic.updateSegmentationClassVisualization(
                self._parameterNode.reconstructedVolume,
                self._parameterNode.showKidney,
                self._parameterNode.showCalyx
            )
        
#
# NeedleGuideLogic
#


class NeedleGuideLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    # transform names
    REFERENCE_TO_RAS = "ReferenceToRas"
    IMAGE_TO_REFERENCE = "ImageToReference"
    PREDICTION_TO_REFERENCE = "PredToReference"
    STYLUS_TO_REFERENCE = "StylusToReference"
    STYLUS_TIP_TO_STYLUS = "StylusTipToStylus"

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

    # needle guide parameters
    NEEDLE_MODEL = "NeedleModel"
    NEEDLE_LENGTH = 80  # mm
    NEEDLE_GUIDE_MARKUP = "NeedleGuideMarkup"

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)
        
        self.reconstructing = False
        self.inPlaneOverlayModelNode = None

    def getParameterNode(self):
        return NeedleGuideParameterNode(super().getParameterNode())

    def setup(self):
        # create nodes for image, prediction, volume reconstruction, and transforms
        parameterNode = self.getParameterNode()
        if not parameterNode.referenceToRas:
            referenceToRas = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", self.REFERENCE_TO_RAS)
            parameterNode.referenceToRas = referenceToRas
        
        if not parameterNode.imageToReference:
            imageToReference = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", self.IMAGE_TO_REFERENCE)
            imageToReference.SetAndObserveTransformNodeID(parameterNode.referenceToRas.GetID())
            parameterNode.imageToReference = imageToReference

        if not parameterNode.inputVolume:
            inputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", self.IMAGE_IMAGE)
            inputVolume.CreateDefaultDisplayNodes()
            inputArray = np.zeros((1, 512, 512), dtype="uint8")
            slicer.util.updateVolumeFromArray(inputVolume, inputArray)
            inputVolume.SetAndObserveTransformNodeID(parameterNode.imageToReference.GetID())
            parameterNode.inputVolume = inputVolume

        if not parameterNode.predictionToReference:
            predictionToReference = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", self.PREDICTION_TO_REFERENCE)
            predictionToReference.SetAndObserveTransformNodeID(parameterNode.referenceToRas.GetID())
            parameterNode.predictionToReference = predictionToReference
        
        if not parameterNode.predictionVolume:
            predictionVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", self.PREDICTION)
            predictionVolume.CreateDefaultDisplayNodes()
            predictionArray = np.zeros((1, 512, 512), dtype="uint8")
            slicer.util.updateVolumeFromArray(predictionVolume, predictionArray)
            predictionVolume.SetAndObserveTransformNodeID(parameterNode.predictionToReference.GetID())
            parameterNode.predictionVolume = predictionVolume

        if not parameterNode.stylusToReference:
            stylusToReference = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", self.STYLUS_TO_REFERENCE)
            stylusToReference.SetAndObserveTransformNodeID(parameterNode.referenceToRas.GetID())
            parameterNode.stylusToReference = stylusToReference

        if not parameterNode.stylusTipToStylus:
            stylusTipToStylus = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", self.STYLUS_TIP_TO_STYLUS)
            stylusTipToStylus.SetAndObserveTransformNodeID(parameterNode.stylusToReference.GetID())
            # TODO: update with actual stylus tip to stylus transform, probably best to save as a .h5 file
            parameterNode.stylusTipToStylus = stylusTipToStylus
        
        if not parameterNode.needleModel:
            createModelsLogic = slicer.modules.createmodels.logic()
            needleModel = createModelsLogic.CreateNeedle(self.NEEDLE_LENGTH, 1.0, 2.5, False)
            needleModel.GetDisplayNode().SetColor(0.33, 1.0, 1.0)
            needleModel.SetName(self.NEEDLE_MODEL)
            needleModel.GetDisplayNode().Visibility2DOn()
            needleModel.SetAndObserveTransformNodeID(parameterNode.stylusTipToStylus.GetID())
            parameterNode.needleModel = needleModel

        if not parameterNode.reconstructedVolume:
            reconstructedVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", self.RECONSTRUCTED_VOLUME)
            reconstructedVolume.CreateDefaultDisplayNodes()
            volRenLogic = slicer.modules.volumerendering.logic()
            displayNode = volRenLogic.CreateDefaultVolumeRenderingNodes(reconstructedVolume)
            displayNode.SetVisibility(True)
            displayNode.GetVolumePropertyNode().Copy(volRenLogic.GetPresetByName("MR-Default"))
            parameterNode.reconstructedVolume = reconstructedVolume

        if not parameterNode.reconstructorNode:
            reconstructorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLVolumeReconstructionNode", self.RECONSTRUCTOR_NODE)
            reconstructorNode.SetLiveVolumeReconstruction(True)
            reconstructorNode.SetInterpolationMode(1)  # linear
            reconstructorNode.SetAndObserveInputVolumeNode(parameterNode.predictionVolume)
            reconstructorNode.SetAndObserveOutputVolumeNode(parameterNode.reconstructedVolume)

            # create roi node
            roiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", self.RECONSTRUCTION_ROI)
            roiNode.SetSize((250, 250, 350))
            roiNode.SetDisplayVisibility(False)
            reconstructorNode.SetAndObserveInputROINode(roiNode)
            parameterNode.reconstructorNode = reconstructorNode

        if not parameterNode.needleGuideMarkups:
            needleGuideMarkups = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", self.NEEDLE_GUIDE_MARKUP)
            needleGuideMarkups.SetMaximumNumberOfControlPoints(1)
            needleGuideMarkups.CreateDefaultDisplayNodes()
            needleGuideMarkups.SetDisplayVisibility(False)
            parameterNode.needleGuideMarkups = needleGuideMarkups
        
        self.setupOpenIgtLink()
    
    def setupOpenIgtLink(self):
        parameterNode = self.getParameterNode()

        # create OpenIGTLink connector node for ultrasound image and tracking
        plusConnectorNode = parameterNode.plusConnectorNode
        if not plusConnectorNode:
            plusConnectorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLIGTLConnectorNode", self.PLUS_CONNECTOR)
            plusConnectorNode.SetTypeClient("localhost", self.PLUS_CONNECTOR_PORT)
            parameterNode.plusConnectorNode = plusConnectorNode
        
        # create OpenIGTLink connector node for prediction
        predictionConnectorNode = parameterNode.predictionConnectorNode
        if not predictionConnectorNode:
            predictionConnectorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLIGTLConnectorNode", self.PREDICTION_CONNECTOR)
            predictionConnectorNode.SetTypeClient("localhost", self.PREDICTION_CONNECTOR_PORT)
            parameterNode.predictionConnectorNode = predictionConnectorNode
    
    def startVolumeReconstruction(self):
        """
        Start live volume reconstruction.
        """
        parameterNode = self.getParameterNode()
        self.reconstructing = True
        reconstructionLogic = slicer.modules.volumereconstruction.logic()
        reconstructionLogic.StartLiveVolumeReconstruction(parameterNode.reconstructorNode)
        outputVolume = parameterNode.reconstructorNode.GetOutputVolumeNode()
        # Use segmentation class visualization if it's a segmentation volume
        # Otherwise use standard volume rendering
        if parameterNode.showKidney or parameterNode.showCalyx:
            self.updateSegmentationClassVisualization(
                outputVolume,
                parameterNode.showKidney,
                parameterNode.showCalyx
            )
        else:
            self.setVolumeRenderingProperty(outputVolume, window=200, level=(255-parameterNode.opacityThreshold))
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
    
    def updateSegmentationClassVisualization(self, volumeNode, showKidney=True, showCalyx=True):
        """
        Update volume rendering to show selected segmentation classes with different colors.
        
        Classes:
        - 0: Background (always transparent)
        - 1: Kidney (red/brown)
        - 2: Calyx (blue/cyan)
        """
        if not volumeNode:
            return
        
        volumeRenderingLogic = slicer.modules.volumerendering.logic()
        volumeRenderingDisplayNode = volumeRenderingLogic.GetFirstVolumeRenderingDisplayNode(volumeNode)
        if not volumeRenderingDisplayNode:
            volumeRenderingDisplayNode = volumeRenderingLogic.CreateDefaultVolumeRenderingNodes(volumeNode)
        
        # Create color transfer function for segmentation classes
        colorTransferFunction = vtk.vtkColorTransferFunction()
        
        # Create opacity transfer function
        opacityTransferFunction = vtk.vtkPiecewiseFunction()
        
        # Class 0: Background - Always transparent (never shown)
        colorTransferFunction.AddRGBPoint(0, 0.0, 0.0, 0.0)  # Black
        opacityTransferFunction.AddPoint(0, 0.0)  # Fully transparent
        
        # Class 1: Kidney - Red/Brown
        if showKidney:
            colorTransferFunction.AddRGBPoint(1, 0.8, 0.2, 0.2)  # Red
            opacityTransferFunction.AddPoint(1, 0.8)  # Mostly opaque
        else:
            colorTransferFunction.AddRGBPoint(1, 0.0, 0.0, 0.0)  # Black
            opacityTransferFunction.AddPoint(1, 0.0)  # Fully transparent
        
        # Class 2: Calyx - Blue/Cyan
        if showCalyx:
            colorTransferFunction.AddRGBPoint(2, 0.2, 0.6, 0.9)  # Cyan/Blue
            opacityTransferFunction.AddPoint(2, 0.9)  # Mostly opaque
        else:
            colorTransferFunction.AddRGBPoint(2, 0.0, 0.0, 0.0)  # Black
            opacityTransferFunction.AddPoint(2, 0.0)  # Fully transparent
        
        # Set smooth transitions between classes
        colorTransferFunction.AddRGBPoint(0.5, 0.4, 0.4, 0.4)  # Transition between 0 and 1
        colorTransferFunction.AddRGBPoint(1.5, 0.5, 0.4, 0.5)  # Transition between 1 and 2
        
        # Apply to volume property
        volumeProperty = volumeRenderingDisplayNode.GetVolumePropertyNode().GetVolumeProperty()
        volumeProperty.SetColor(colorTransferFunction)
        volumeProperty.SetScalarOpacity(opacityTransferFunction)
        volumeProperty.ShadeOn()
        volumeProperty.SetInterpolationTypeToLinear()
        
        # Ensure visibility is on
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
        
    def showInPlaneDepthLines(self):
        if not self.inPlaneOverlayModelNode:
            self.inPlaneOverlayModelNode = slicer.mrmlScene.AddNewNodeByClass(
                'vtkMRMLModelNode', 'InPlaneDepthOverlayModel')

            self._createInPlaneDepthFanModel(self.inPlaneOverlayModelNode)

        self.inPlaneOverlayModelNode.SetDisplayVisibility(True)

    def hideInPlaneDepthLines(self):
        if self.inPlaneOverlayModelNode:
            self.inPlaneOverlayModelNode.SetDisplayVisibility(False)

    def _createInPlaneDepthLinesModel(self, modelNode):
    
        imageNode = slicer.util.getNode('Image_Image')
        imageData = imageNode.GetImageData()
        spacing = imageNode.GetSpacing()      # (x_spacing, y_spacing, z_spacing)
        origin = imageNode.GetOrigin()        # (x0, y0, z0)
        dims = imageData.GetDimensions()      # (cols, rows, slices)

        apex = [origin[0] + (dims[0] - 1) * spacing[0],  # rightmost X
                origin[1] + (dims[1] - 1) * spacing[1],  # top Y
                origin[2]                                # Z slice
            ]

        # 📐 Use CAD-defined entry angles and depths
        angles_deg = [44.8, 31.3, 24.1]
        depths_mm = [45.0, 76.304, 93.166]

        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        pointId = 0
        scale = 1.0 / 0.189

        for angle_deg, depth in zip(angles_deg, depths_mm):
            angle_rad = math.radians(angle_deg)

            dx = -depth * math.sin(angle_rad) * scale   # → right
            dy = -depth * math.cos(angle_rad) * scale  # ↓ down (in image)

            p0 = apex
            p1 = [apex[0] + dx, apex[1] + dy, apex[2]]

            points.InsertNextPoint(p0)
            points.InsertNextPoint(p1)

            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, pointId)
            line.GetPointIds().SetId(1, pointId + 1)
            lines.InsertNextCell(line)
            pointId += 2

        # 🔧 Create and assign polydata
        polyData = vtk.vtkPolyData()
        polyData.SetPoints(points)
        polyData.SetLines(lines)
        modelNode.SetAndObservePolyData(polyData)

        # 🎨 Display node
        displayNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelDisplayNode')
        slicer.mrmlScene.AddNode(displayNode)
        modelNode.SetAndObserveDisplayNodeID(displayNode.GetID())
        displayNode.SetColor(0.0, 1.0, 0.0)
        displayNode.SetLineWidth(2)
        displayNode.VisibilityOn()
        displayNode.SetSliceIntersectionVisibility(True)

        # 🔗 Transform to reference space
        imageToReference = slicer.util.getNode('ImageToReference')
        if imageToReference:
            modelNode.SetAndObserveTransformNodeID(imageToReference.GetID())

    def _createInPlaneDepthFanModel(self, modelNode):
  
        # Get the ultrasound image node
        imageNode = slicer.util.getNode('Image_Image')
        imageData = imageNode.GetImageData()
        spacing = imageNode.GetSpacing()
        origin = imageNode.GetOrigin()
        dims = imageData.GetDimensions()

        # Determine the fan apex in IMAGE space:
        # Apex: 2D origin point from which the curvilinear beams radiate,
        # located at the center of the top edge of the image.
        apexX = origin[0] + (dims[0] - 1) / 2.0 * spacing[0]  # center column
        apexY = origin[1]        # top row (flip Y)
        apexZ = origin[2]                                    # same slice

        # CAD-defined in-plane channels (depths in mm, angles in degrees)
        depths_mm = [45.0, 76.304, 93.166]
        angles_deg = [44.8, 31.3, 24.1]
        # Lateral offsets of each channel hole from probe midline (in mm)
        # Adjust these to match your needle guide design
        channelOffsets_mm = [-10.0, 0.0, 10.0]

        # Compute calibration scale (mm→image units) from ImageToProbe matrix
        imageToProbeNode = slicer.util.getNode('ImageToProbe', False) or slicer.util.getNode('ImageToPobe')
        imageToProbeMatrix = vtk.vtkMatrix4x4()
        imageToProbeNode.GetMatrixTransformToParent(imageToProbeMatrix)
        sx = imageToProbeMatrix.GetElement(0,0)
        sy = imageToProbeMatrix.GetElement(1,1)
        sz = imageToProbeMatrix.GetElement(2,2)
        avgScale = (abs(sx) + abs(sy) + abs(sz)) / 3.0
        mmToImageScale = 1.0 / avgScale

        # Build VTK points and lines for each channel
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        pointId = 0
        for offset_mm, depth_mm, angle_deg in zip(channelOffsets_mm, depths_mm, angles_deg):
            angle_rad = math.radians(angle_deg)
            # Compute channel start in image space: shift apex laterally
            offset_image = offset_mm * mmToImageScale
            p0 = [apexX + offset_image, apexY, apexZ]
            # Compute channel direction vector (left/right offset already in p0) (left/right offset already in p0)
            dx = depth_mm * math.sin(angle_rad) * mmToImageScale
            dy = depth_mm * math.cos(angle_rad) * mmToImageScale
            p1 = [p0[0] + dx, p0[1] + dy, p0[2]]
            points.InsertNextPoint(p0)
            points.InsertNextPoint(p1)
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, pointId)
            line.GetPointIds().SetId(1, pointId + 1)
            lines.InsertNextCell(line)
            pointId += 2

        # Create polydata and assign to model node
        polyData = vtk.vtkPolyData()
        polyData.SetPoints(points)
        polyData.SetLines(lines)
        modelNode.SetAndObservePolyData(polyData)
        polyData = vtk.vtkPolyData()
        polyData.SetPoints(points)
        polyData.SetLines(lines)
        modelNode.SetAndObservePolyData(polyData)

        # Create and configure display node
        # (after polydata has been assigned to modelNode)
        displayNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelDisplayNode')
        modelNode.SetAndObserveDisplayNodeID(displayNode.GetID())

        # styling
        displayNode.SetColor(0.0, 1.0, 0.0)
        displayNode.SetLineWidth(3)

        # --- critical for 2D slice views ---
        displayNode.SetVisibility2D(True)                          # show in slices :contentReference[oaicite:3]{index=3}
        displayNode.SetSliceDisplayModeToIntersection()            # draw only intersection lines :contentReference[oaicite:4]{index=4}
        displayNode.SetSliceIntersectionThickness(2)

        # force all slice views to update and draw intersections
        for sd in slicer.util.getNodesByClass('vtkMRMLSliceDisplayNode'):
            sd.SetIntersectingSlicesVisibility(1)
        for sn in slicer.util.getNodesByClass('vtkMRMLSliceNode'):
            sn.Modified()
        # Parent the overlay to the same transform as the image (follows probe motion)
        parentTransform = imageNode.GetParentTransformNode()
        if parentTransform:
            modelNode.SetAndObserveTransformNodeID(parentTransform.GetID())



#
# NeedleGuideTest
#


class NeedleGuideTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_NeedleGuide1()

    def test_NeedleGuide1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("NeedleGuide1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = NeedleGuideLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")
