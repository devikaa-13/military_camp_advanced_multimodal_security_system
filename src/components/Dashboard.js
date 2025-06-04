import React, { useState, useEffect, useCallback } from 'react';
import { Grid, Paper, Typography, Box } from '@mui/material';
import SurveillanceFeed from './SurveillanceFeed';
import EventLog from './EventLog';
import ThreatAlerts from './ThreatAlerts';
import DataManager from './DataManager';

// --- Simulation Data ---
const CAMERAS = ['CAM_01_NORTH_GATE', 'CAM_02_PERIMETER_WEST', 'CAM_03_BARRACKS', 'CAM_04_MOTORPOOL'];
const YOLO_CLASSES = ['weapon', 'soldier', 'civilian', 'vehicle'];
const YAMNET_SOUNDS = ['Speech', 'Vehicle noise', 'Gunshot', 'Explosion', 'Silence', 'Alarm'];
const SOLDIER_IDS = ['S101', 'S102', 'S103', 'S104', 'S105']; // Verified soldiers
const VEHICLE_PLATES = ['MH-01-AB-1234', 'KA-05-CD-5678', 'DL-03-EF-9012']; // Verified vehicles

// --- Main Dashboard Component ---
const Dashboard = () => {
  const [events, setEvents] = useState([]);
  const [threats, setThreats] = useState([]);
  const [soldiers, setSoldiers] = useState(SOLDIER_IDS.map(id => ({ id, name: `Soldier ${id}` }))); // Example structure
  const [vehicles, setVehicles] = useState(VEHICLE_PLATES.map(plate => ({ plate, description: `Vehicle ${plate}` }))); // Example structure

  // --- Data Simulation Logic ---
  const generateSimulatedEvent = useCallback(() => {
    const timestamp = new Date();
    const camera = CAMERAS[Math.floor(Math.random() * CAMERAS.length)];
    const eventType = Math.random();

    let newEvent = null;
    let newThreat = null;

    if (eventType < 0.4) { // YOLO Detection
      const detectedClass = YOLO_CLASSES[Math.floor(Math.random() * YOLO_CLASSES.length)];
      const isNearCivilian = detectedClass === 'weapon' && Math.random() < 0.3; // Simulate weapon near civilian

      newEvent = {
        id: timestamp.getTime() + Math.random(),
        timestamp,
        type: 'YOLO',
        camera,
        details: `Detected: ${detectedClass}`,
        severity: 'info',
      };

      if (detectedClass === 'civilian') {
        newThreat = { ...newEvent, type: 'THREAT_CIVILIAN', severity: 'warning', details: `Civilian detected near ${camera}` };
        newEvent.severity = 'warning';
      } else if (isNearCivilian) {
        newThreat = { ...newEvent, type: 'THREAT_WEAPON_CIVILIAN', severity: 'error', details: `Weapon detected near civilian at ${camera}` };
        newEvent.severity = 'error';
      } else if (detectedClass === 'weapon') {
        newEvent.severity = 'warning';
      } else if (detectedClass === 'vehicle') {
          const plateDetected = Math.random() < 0.7; // Simulate OCR success
          if (plateDetected) {
              const detectedPlate = Math.random() < 0.8 ? vehicles[Math.floor(Math.random() * vehicles.length)].plate : `NEW-${Math.random().toString(36).substring(2, 8).toUpperCase()}`;
              const isVerified = vehicles.some(v => v.plate === detectedPlate);
              newEvent.details += ` | Plate: ${detectedPlate} (${isVerified ? 'Verified' : 'UNVERIFIED'})`;
              if (!isVerified) {
                  newEvent.severity = 'error';
                  newThreat = { ...newEvent, type: 'THREAT_VEHICLE', severity: 'error', details: `Unverified vehicle (${detectedPlate}) detected at ${camera}` };
              } else {
                  newEvent.severity = 'success';
              }
          } else {
              newEvent.details += ' | Plate OCR Failed';
              newEvent.severity = 'warning';
          }
      } else if (detectedClass === 'soldier') {
          const faceRecognized = Math.random() < 0.8; // Simulate face rec success
          if (faceRecognized) {
              const detectedSoldierId = Math.random() < 0.8 ? soldiers[Math.floor(Math.random() * soldiers.length)].id : `UNKNOWN-${Math.random().toString(10).substring(2, 6)}`;
              const isVerified = soldiers.some(s => s.id === detectedSoldierId);
               newEvent.details += ` | Face ID: ${detectedSoldierId} (${isVerified ? 'Verified' : 'UNVERIFIED'})`;
              if (!isVerified) {
                  newEvent.severity = 'error';
                   newThreat = { ...newEvent, type: 'THREAT_SOLDIER', severity: 'error', details: `Unverified person (${detectedSoldierId}) detected near ${camera}` };
              } else {
                 newEvent.severity = 'success';
              }
          } else {
             newEvent.details += ' | Face Recognition Failed';
             newEvent.severity = 'warning';
          }
      }

    } else if (eventType < 0.6) { // YamNet Detection
      const sound = YAMNET_SOUNDS[Math.floor(Math.random() * YAMNET_SOUNDS.length)];
      const severity = (sound === 'Gunshot' || sound === 'Explosion' || sound === 'Alarm') ? 'warning' : 'info';
      newEvent = {
        id: timestamp.getTime() + Math.random(),
        timestamp,
        type: 'Audio',
        camera, // Assuming mics are near cameras
        details: `Sound detected: ${sound}`,
        severity: severity,
      };
       if (severity === 'warning') {
          newThreat = { ...newEvent, type: 'THREAT_AUDIO', severity: 'warning', details: `Suspicious sound (${sound}) detected near ${camera}` };
       }
    } else { // Qwen Description (simulated)
      newEvent = {
        id: timestamp.getTime() + Math.random(),
        timestamp,
        type: 'Qwen-VL',
        camera,
        details: `Scene at ${camera}: ${Math.random() < 0.5 ? 'Routine patrol activity.' : 'Vehicle movement detected.'} Ambient sound level normal.`,
        severity: 'info',
      };
    }

    if (newEvent) {
      setEvents(prevEvents => [newEvent, ...prevEvents].slice(0, 100)); // Keep last 100 events
    }
    if (newThreat) {
       // Avoid duplicate threats for the same root cause quickly
       setThreats(prevThreats => {
           const similarThreatExists = prevThreats.some(t =>
               t.type === newThreat.type && t.camera === newThreat.camera && (timestamp.getTime() - t.timestamp.getTime()) < 30000 // Within 30s
           );
           if (!similarThreatExists) {
               return [newThreat, ...prevThreats].slice(0, 20); // Keep last 20 threats
           }
           return prevThreats;
       });
    }
  }, [vehicles, soldiers]); // Depend on databases for verification checks

  // Simulate real-time updates
  useEffect(() => {
    const intervalId = setInterval(generateSimulatedEvent, 2000); // Generate a new event every 2 seconds
    return () => clearInterval(intervalId); // Cleanup on unmount
  }, [generateSimulatedEvent]);

  // --- Callback Handlers for DataManager ---
  const handleAddSoldier = (soldier) => {
      setSoldiers(prev => [...prev, { ...soldier, id: soldier.id || `S${Math.random().toString(10).substring(2,6)}` }]); // Add or generate ID
  };

  const handleDeleteSoldier = (soldierId) => {
      setSoldiers(prev => prev.filter(s => s.id !== soldierId));
  };

   const handleAddVehicle = (vehicle) => {
      setVehicles(prev => [...prev, vehicle]);
  };

  const handleDeleteVehicle = (vehiclePlate) => {
      setVehicles(prev => prev.filter(v => v.plate !== vehiclePlate));
  };

  const acknowledgeThreat = (threatId) => {
      setThreats(prevThreats => prevThreats.filter(t => t.id !== threatId));
  }

  return (
    <Grid container spacing={3}>
      {/* Threat Alerts - Prominent */}
      <Grid item xs={12}>
         <ThreatAlerts threats={threats} onAcknowledge={acknowledgeThreat} />
      </Grid>

      {/* Surveillance Feeds */}
      {CAMERAS.map(camId => (
        <Grid item xs={12} md={6} lg={3} key={camId}>
          <SurveillanceFeed cameraId={camId} latestEvents={events.filter(e => e.camera === camId).slice(0, 3)} />
        </Grid>
      ))}

       {/* Event Log */}
      <Grid item xs={12} lg={8}>
         <EventLog events={events} />
      </Grid>

       {/* Data Management */}
      <Grid item xs={12} lg={4}>
          <DataManager
              soldiers={soldiers}
              vehicles={vehicles}
              onAddSoldier={handleAddSoldier}
              onDeleteSoldier={handleDeleteSoldier}
              onAddVehicle={handleAddVehicle}
              onDeleteVehicle={handleDeleteVehicle}
           />
      </Grid>

    </Grid>
  );
};

export default Dashboard;