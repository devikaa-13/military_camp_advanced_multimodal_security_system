import React, { useState } from 'react';
import {
  Card, CardContent, Typography, Box, Button, List, ListItem, ListItemText, IconButton,
  Modal, TextField, Paper, Divider, Tabs, Tab
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import AddCircleOutlineIcon from '@mui/icons-material/AddCircleOutline';
import GroupIcon from '@mui/icons-material/Group';
import DirectionsCarIcon from '@mui/icons-material/DirectionsCar';
import StorageIcon from '@mui/icons-material/Storage';

const modalStyle = {
  position: 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  width: 400,
  bgcolor: 'background.paper',
  border: '2px solid #000',
  boxShadow: 24,
  p: 4,
};

function TabPanel(props) {
  const { children, value, index, ...other } = props;
  return (
    <div role="tabpanel" hidden={value !== index} id={`simple-tabpanel-${index}`} aria-labelledby={`simple-tab-${index}`} {...other}>
      {value === index && <Box sx={{ pt: 2 }}>{children}</Box>}
    </div>
  );
}

const DataManager = ({ soldiers, vehicles, onAddSoldier, onDeleteSoldier, onAddVehicle, onDeleteVehicle }) => {
  const [soldierModalOpen, setSoldierModalOpen] = useState(false);
  const [vehicleModalOpen, setVehicleModalOpen] = useState(false);
  const [newSoldierName, setNewSoldierName] = useState('');
  const [newSoldierId, setNewSoldierId] = useState(''); // Optional: Allow manual ID entry
  const [newVehiclePlate, setNewVehiclePlate] = useState('');
  const [newVehicleDesc, setNewVehicleDesc] = useState('');
  const [tabValue, setTabValue] = React.useState(0);

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const handleAddSoldierClick = () => {
    if (newSoldierName.trim()) { // Basic validation
      onAddSoldier({ name: newSoldierName, id: newSoldierId.trim() || undefined }); // Pass undefined if ID is empty
      setNewSoldierName('');
      setNewSoldierId('');
      setSoldierModalOpen(false);
    }
  };

   const handleAddVehicleClick = () => {
    if (newVehiclePlate.trim()) { // Basic validation
      onAddVehicle({ plate: newVehiclePlate, description: newVehicleDesc });
      setNewVehiclePlate('');
      setNewVehicleDesc('');
      setVehicleModalOpen(false);
    }
  };

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
         <Box display="flex" alignItems="center" mb={2}>
            <StorageIcon sx={{mr: 1}}/>
            <Typography variant="h6">Database Management</Typography>
         </Box>

        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={tabValue} onChange={handleTabChange} aria-label="database tabs">
            <Tab icon={<GroupIcon />} iconPosition="start" label="Soldiers" id="simple-tab-0" aria-controls="simple-tabpanel-0" />
            <Tab icon={<DirectionsCarIcon />} iconPosition="start" label="Vehicles" id="simple-tab-1" aria-controls="simple-tabpanel-1" />
          </Tabs>
        </Box>

        {/* Soldiers Tab */}
        <TabPanel value={tabValue} index={0}>
           <Button
              variant="contained"
              startIcon={<AddCircleOutlineIcon />}
              onClick={() => setSoldierModalOpen(true)}
              size="small"
              sx={{ mb: 1 }}
           >
            Add Soldier
           </Button>
           <List dense sx={{ maxHeight: 300, overflow: 'auto' }}>
             {soldiers.map((soldier) => (
               <ListItem
                 key={soldier.id}
                 secondaryAction={
                   <IconButton edge="end" aria-label="delete" onClick={() => onDeleteSoldier(soldier.id)}>
                     <DeleteIcon />
                   </IconButton>
                 }
                 divider
               >
                 <ListItemText primary={soldier.name} secondary={`ID: ${soldier.id}`} />
               </ListItem>
             ))}
           </List>
        </TabPanel>

        {/* Vehicles Tab */}
         <TabPanel value={tabValue} index={1}>
             <Button
                variant="contained"
                startIcon={<AddCircleOutlineIcon />}
                onClick={() => setVehicleModalOpen(true)}
                size="small"
                sx={{ mb: 1 }}
             >
                Add Vehicle
             </Button>
             <List dense sx={{ maxHeight: 300, overflow: 'auto' }}>
                {vehicles.map((vehicle) => (
                 <ListItem
                    key={vehicle.plate}
                    secondaryAction={
                    <IconButton edge="end" aria-label="delete" onClick={() => onDeleteVehicle(vehicle.plate)}>
                        <DeleteIcon />
                    </IconButton>
                    }
                    divider
                 >
                    <ListItemText primary={vehicle.plate} secondary={vehicle.description} />
                 </ListItem>
                ))}
             </List>
        </TabPanel>

        {/* Add Soldier Modal */}
        <Modal open={soldierModalOpen} onClose={() => setSoldierModalOpen(false)}>
          <Paper sx={modalStyle}>
            <Typography variant="h6" component="h2" mb={2}> Add New Soldier </Typography>
            <TextField label="Soldier Name" value={newSoldierName} onChange={(e) => setNewSoldierName(e.target.value)} fullWidth margin="normal" required />
            <TextField label="Soldier ID (Optional)" value={newSoldierId} onChange={(e) => setNewSoldierId(e.target.value)} fullWidth margin="normal" helperText="Leave blank for auto-generation" />
             {/* In reality, you'd have a way to upload/link face data here */}
            <Box mt={2} display="flex" justifyContent="flex-end">
               <Button onClick={() => setSoldierModalOpen(false)} sx={{ mr: 1 }}>Cancel</Button>
               <Button variant="contained" onClick={handleAddSoldierClick}>Add Soldier</Button>
            </Box>
          </Paper>
        </Modal>

        {/* Add Vehicle Modal */}
        <Modal open={vehicleModalOpen} onClose={() => setVehicleModalOpen(false)}>
          <Paper sx={modalStyle}>
            <Typography variant="h6" component="h2" mb={2}> Add New Vehicle </Typography>
            <TextField label="Vehicle Plate Number" value={newVehiclePlate} onChange={(e) => setNewVehiclePlate(e.target.value)} fullWidth margin="normal" required/>
            <TextField label="Description (e.g., Make/Model)" value={newVehicleDesc} onChange={(e) => setNewVehicleDesc(e.target.value)} fullWidth margin="normal" />
            <Box mt={2} display="flex" justifyContent="flex-end">
               <Button onClick={() => setVehicleModalOpen(false)} sx={{ mr: 1 }}>Cancel</Button>
               <Button variant="contained" onClick={handleAddVehicleClick}>Add Vehicle</Button>
            </Box>
          </Paper>
        </Modal>

      </CardContent>
    </Card>
  );
};

export default DataManager;