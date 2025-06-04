import React from 'react';
import { Card, CardContent, Typography, Box, Chip } from '@mui/material';
import VideocamIcon from '@mui/icons-material/Videocam';
import InfoIcon from '@mui/icons-material/Info';
import WarningIcon from '@mui/icons-material/Warning';
import ErrorIcon from '@mui/icons-material/Error';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';

const getIcon = (severity) => {
    switch (severity) {
        case 'error': return <ErrorIcon color="error" fontSize="small" />;
        case 'warning': return <WarningIcon color="warning" fontSize="small" />;
        case 'success': return <CheckCircleIcon color="success" fontSize="small" />;
        case 'info':
        default: return <InfoIcon color="info" fontSize="small" />;
    }
};

const SurveillanceFeed = ({ cameraId, latestEvents }) => {
  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box display="flex" alignItems="center" mb={2}>
          <VideocamIcon sx={{ mr: 1 }} />
          <Typography variant="h6" component="div">
            {cameraId}
          </Typography>
        </Box>
        {/* Placeholder for actual video feed */}
        <Box
          sx={{
            height: 150,
            backgroundColor: 'rgba(255, 255, 255, 0.1)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            borderRadius: 1,
            mb: 2,
          }}
        >
          <Typography variant="caption" color="text.secondary">
            Live Feed Area (Simulated)
          </Typography>
        </Box>
         <Typography variant="subtitle2" gutterBottom>Latest Events:</Typography>
         {latestEvents.length > 0 ? (
             latestEvents.map(event => (
                 <Chip
                    key={event.id}
                    icon={getIcon(event.severity)}
                    label={`${event.type}: ${event.details.substring(0, 40)}${event.details.length > 40 ? '...' : ''}`}
                    size="small"
                    variant="outlined"
                    color={event.severity === 'error' ? 'error' : event.severity === 'warning' ? 'warning' : event.severity === 'success' ? 'success' : 'info'}
                    sx={{ mr: 0.5, mb: 0.5 }}
                    title={event.details} // Show full details on hover
                 />
             ))
         ) : (
             <Typography variant="body2" color="text.secondary">No recent events.</Typography>
         )}
      </CardContent>
    </Card>
  );
};

export default SurveillanceFeed;