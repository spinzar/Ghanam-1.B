class CustomFooter extends HTMLElement {
  connectedCallback() {
    this.attachShadow({ mode: 'open' });
    this.shadowRoot.innerHTML = `
      <style>
        :host {
          display: block;
          margin-top: 4rem;
        }
        
        .footer {
          background-color: #1f2937;
          color: white;
          padding: 3rem 0 2rem;
        }
        
        .container {
          max-width: 1200px;
          margin: 0 auto;
          padding: 0 1rem;
        }
        
        .footer-content {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 2rem;
          margin-bottom: 2rem;
        }
        
        .footer-section h3 {
          font-size: 1.25rem;
          margin-bottom: 1rem;
          color: #60a5fa;
        }
        
        .footer-section ul {
          list-style: none;
          padding: 0;
        }
        
        .footer-section ul li {
          margin-bottom: 0.5rem;
        }
        
        .footer-section ul li a {
          color: #d1d5db;
          text-decoration: none;
          transition: color 0.3s ease;
        }
        
        .footer-section ul li a:hover {
          color: white;
        }
        
        .social-links {
          display: flex;
          gap: 1rem;
          margin-top: 1rem;
        }
        
        .social-links a {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          width: 40px;
          height: 40px;
          background-color: #374151;
          border-radius: 50%;
          color: white;
          transition: background-color 0.3s ease;
        }
        
        .social-links a:hover {
          background-color: #60a5fa;
        }
        
        .copyright {
          text-align: center;
          padding-top: 2rem;
          border-top: 1px solid #374151;
          color: #9ca3af;
          font-size: 0.875rem;
        }
      </style>
      
      <footer class="footer">
        <div class="container">
          <div class="footer-content">
            <div class="footer-section">
              <h3>Ghanam-1.B</h3>
              <p>A community-driven initiative to build the first open-source Large Language Model for Pashto language.</p>
              <div class="social-links">
                <a href="#"><i data-feather="github"></i></a>
                <a href="#"><i data-feather="twitter"></i></a>
                <a href="#"><i data-feather="message-circle"></i></a>
              </div>
            </div>
            
            <div class="footer-section">
              <h3>Resources</h3>
              <ul>
                <li><a href="#">Documentation</a></li>
                <li><a href="#">GitHub Repository</a></li>
                <li><a href="#">Contribution Guide</a></li>
                <li><a href="#">Roadmap</a></li>
              </ul>
            </div>
            
            <div class="footer-section">
              <h3>Community</h3>
              <ul>
                <li><a href="#">Discord</a></li>
                <li><a href="#">Mailing List</a></li>
                <li><a href="#">Events</a></li>
                <li><a href="#">Code of Conduct</a></li>
              </ul>
            </div>
          </div>
          
          <div class="copyright">
            <p>&copy; 2023 Ghanam-1.B Initiative. Open source under MIT License.</p>
          </div>
        </div>
      </footer>
    `;
  }
}

customElements.define('custom-footer', CustomFooter);