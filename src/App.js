import React from 'react';
import { ThemeProvider, createTheme, CssBaseline, Box, AppBar, Toolbar, Typography } from '@mui/material';
import SecurityIcon from '@mui/icons-material/Security';
import Dashboard from './components/Dashboard';

// Define a theme (customize colors as needed)
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#90caf9', // Light blue
    },
    secondary: {
      main: '#f48fb1', // Pink
    },
    background: {
      default: '#121212',
      paper: '#1e1e1e',
    },
    error: {
      main: '#f44336', // Red
    },
    warning: {
      main: '#ffa726', // Orange
    },
    info: {
      main: '#29b6f6', // Blue
    },
    success: {
      main: '#66bb6a', // Green
    },
  },
  typography: {
    fontFamily: 'Roboto, Arial, sans-serif',
    h5: {
      fontWeight: 600,
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline /> {/* Ensures consistent baseline styling */}
      <Box sx={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
        <AppBar position="static">
          <Toolbar>
            <SecurityIcon sx={{ mr: 2 }} />
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              Military Camp Security Command Center
            </Typography>
            {/* Add any other AppBar items here if needed */}
          </Toolbar>
        </AppBar>
        <Box component="main" sx={{ flexGrow: 1, p: 3, overflow: 'auto' }}>
          <Dashboard />
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;