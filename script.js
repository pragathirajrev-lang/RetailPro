// script.js
// Simple front‑end login handling with smooth fade‑in animation

document.addEventListener('DOMContentLoaded', () => {
  const card = document.querySelector('.login-card');
  // Fade‑in effect
  card.style.opacity = '0';
  card.style.transition = 'opacity 0.6s ease-out';
  requestAnimationFrame(() => {
    card.style.opacity = '1';
  });

  const form = document.getElementById('loginForm');
  const errorMsg = document.getElementById('errorMsg');
  const dashboard = document.getElementById('dashboard');

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const email = form.email.value.trim();
    const password = form.password.value;
    errorMsg.textContent = '';

    if (!email || !password) {
      errorMsg.textContent = 'Please fill in both email and password.';
      return;
    }
    // Only Business Head is allowed to login
    const allowedEmail = 'business.head@example.com';
    const allowedPass = 'SecurePass123!';
    if (email === allowedEmail && password === allowedPass) {
      // Successful login – redirect to dashboard page
      window.location.href = 'dashboard.html';
    } else {
      errorMsg.textContent = 'Invalid credentials. Access is restricted to the Business Head.';
    }
  });
});
