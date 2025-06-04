import React from 'react';
import { Card, CardContent, Typography, List, ListItem, ListItemIcon, ListItemText, IconButton, Box, Chip } from '@mui/material';
import ReportProblemIcon from '@mui/icons-material/ReportProblem';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';

const getThreatIcon = (severity) => {
    switch (severity) {
        case 'error': return <ErrorOutlineIcon color="error" />;
        case 'warning': return <WarningAmberIcon color="warning" />;
        default: return <ReportProblemIcon color="info" />;
    }
};

const ThreatAlerts = ({ threats, onAcknowledge }) => {
  return (
    <Card sx={{ backgroundColor: 'rgba(255, 82, 82, 0.1)' }}> {/* Subtle red background */}
      <CardContent>
         <Box display="flex" alignItems="center" mb={2}>
            <ReportProblemIcon color="error" sx={{ mr: 1 }} />
            <Typography variant="h6" color="error">Active Threats</Typography>
         </Box>
        {threats.length === 0 ? (
          <Typography variant="body1" color="text.secondary" align="center" sx={{p: 2}}>
             No active threats. System nominal.
          </Typography>
        ) : (
          <List dense>
            {threats.map((threat) => (
              <ListItem
                key={threat.id}
                divider
                secondaryAction={
                  <IconButton edge="end" aria-label="acknowledge" onClick={() => onAcknowledge(threat.id)} title="Acknowledge Threat">
                    <CheckCircleOutlineIcon />
                  </IconButton>
                }
                 sx={{ backgroundColor: threat.severity === 'error' ? 'rgba(255, 0, 0, 0.2)' : 'rgba(255, 165, 0, 0.2)', mb: 1, borderRadius: 1}}
              >
                <ListItemIcon sx={{ minWidth: '40px' }}>
                   {getThreatIcon(threat.severity)}
                </ListItemIcon>
                <ListItemText
                   primary={
                       <Typography variant="body1" fontWeight="bold">
                           {threat.type.replace('THREAT_', '').replace('_', ' ')} - {threat.camera}
                       </Typography>
                   }
                   secondary={
                       <>
                        <Typography component="span" variant="body2">
                          {threat.details}
                        </Typography>
                         <br/>
                         <Typography component="span" variant="caption" color="text.secondary">
                           {threat.timestamp.toLocaleString()}
                        </Typography>
                       </>
                   }

                />
              </ListItem>
            ))}
          </List>
        )}
      </CardContent>
    </Card>
  );
};

export default ThreatAlerts;