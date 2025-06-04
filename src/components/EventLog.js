import React from 'react';
import { Card, CardContent, Typography, List, ListItem, ListItemIcon, ListItemText, Chip,Box } from '@mui/material';
import InfoIcon from '@mui/icons-material/Info';
import WarningIcon from '@mui/icons-material/Warning';
import ErrorIcon from '@mui/icons-material/Error';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import HistoryIcon from '@mui/icons-material/History';

const getIcon = (severity) => {
  switch (severity) {
    case 'error': return <ErrorIcon color="error" />;
    case 'warning': return <WarningIcon color="warning" />;
    case 'success': return <CheckCircleIcon color="success" />;
    case 'info':
    default: return <InfoIcon color="info" />;
  }
};

const EventLog = ({ events }) => {
  return (
    <Card sx={{ maxHeight: '600px', overflowY: 'auto' }}>
      <CardContent>
        <Box display="flex" alignItems="center" mb={2}>
            <HistoryIcon sx={{mr: 1}}/>
            <Typography variant="h6">Event Log</Typography>
        </Box>
        <List dense>
          {events.map((event) => (
            <ListItem key={event.id} divider>
              <ListItemIcon sx={{ minWidth: '40px' }}>
                 {getIcon(event.severity)}
              </ListItemIcon>
              <ListItemText
                primary={
                    <Typography variant="body2">
                        {event.details}
                    </Typography>
                 }
                secondary={
                  <>
                    <Typography component="span" variant="caption" color="text.secondary">
                      {event.timestamp.toLocaleTimeString()} - {event.camera} -
                    </Typography>
                     <Chip label={event.type} size="small" variant="outlined" sx={{ml: 0.5}}/>
                  </>
                }
              />
            </ListItem>
          ))}
           {events.length === 0 && (
               <ListItem>
                   <ListItemText primary="No events yet." />
               </ListItem>
           )}
        </List>
      </CardContent>
    </Card>
  );
};

export default EventLog;