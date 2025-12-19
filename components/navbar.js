class CustomNavbar extends HTMLElement {
  connectedCallback() {
    this.attachShadow({ mode: 'open' });
    this.shadowRoot.innerHTML = `
      <style>
        :host {
          display: block;
        }
        
        .navbar {
          background-color: white;
          box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
          padding: 1rem 0;
          position: sticky;
          top: 0;
          z-index: 100;
        }
        
        .container {
          max-width: 1200px;
          margin: 0 auto;
          padding: 0 1rem;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }
        
        .logo {
          font-size: 1.5rem;
          font-weight: 700;
          color: #2563eb;
          text-decoration: none;
          display: flex;
          align-items: center;
        }
        
        .logo i {
          margin-right: 0.5rem;
        }
        
        .nav-links {
          display: flex;
          list-style: none;
        }
        
        .nav-links li {
          margin-left: 2rem;
        }
        
        .nav-links a {
          text-decoration: none;
          color: #4b5563;
          font-weight: 500;
          transition: color 0.3s ease;
        }
        
        .nav-links a:hover {
          color: #2563eb;
        }
        
        .mobile-menu-btn {
          display: none;
          background: none;
          border: none;
          font-size: 1.5rem;
          cursor: pointer;
          color: #4b5563;
        }
        
        @media (max-width: 768px) {
          .nav-links {
            display: none;
            flex-direction: column;
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            background: white;
            padding: 1rem;
            box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
          }
          
          .nav-links.active {
            display: flex;
          }
          
          .nav-links li {
            margin: 0.5rem 0;
          }
          
          .mobile-menu-btn {
            display: block;
          }
        }
      </style>
      
      <nav class="navbar">
        <div class="container">
          <a href="#" class="logo">
            <i data-feather="cpu"></i>
            Ghanam-1.B
          </a>
          
          <button class="mobile-menu-btn" id="mobileMenuBtn">
            <i data-feather="menu"></i>
          </button>
          
          <ul class="nav-links" id="navLinks">
            <li><a href="#"><i data-feather="home" class="mr-1"></i> Home</a></li>
            <li><a href="#"><i data-feather="book" class="mr-1"></i> Documentation</a></li>
            <li><a href="#"><i data-feather="github" class="mr-1"></i> GitHub</a></li>
            <li><a href="#"><i data-feather="users" class="mr-1"></i> Community</a></li>
            <li><a href="#" class="btn-primary">Contribute</a></li>
          </ul>
        </div>
      </nav>
    `;
    
    // Add mobile menu toggle
    const mobileMenuBtn = this.shadowRoot.getElementById('mobileMenuBtn');
    const navLinks = this.shadowRoot.getElementById('navLinks');
    
    mobileMenuBtn.addEventListener('click', () => {
      navLinks.classList.toggle('active');
      const menuIcon = mobileMenuBtn.querySelector('i');
      if (navLinks.classList.contains('active')) {
        menuIcon.setAttribute('data-feather', 'x');
      } else {
        menuIcon.setAttribute('data-feather', 'menu');
      }
      feather.replace();
    });
  }
}

customElements.define('custom-navbar', CustomNavbar);